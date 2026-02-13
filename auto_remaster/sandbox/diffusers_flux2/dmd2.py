import torch
import torch.nn.functional as F


class FlowMatchingDMD2Trainer:
    def __init__(self, student_model, teacher_model):
        self.student = student_model.train()  # Генератор (1 шаг)
        self.teacher = teacher_model.eval()  # Учитель (заморожен)
        self.teacher.requires_grad_(False)

    def compute_loss(self, real_images):
        batch_size = real_images.shape[0]
        device = real_images.device

        # ==========================================
        # 1. ГЕНЕРАЦИЯ СТУДЕНТОМ (Forward Pass)
        # ==========================================
        # Генерируем начальный шум z0
        z0 = torch.randn_like(real_images)

        # Студент делает предсказание за 1 шаг.
        # В Flow Matching 1-step генерация это: x_1 = x_0 + v(x_0, 0) * 1.0
        # То есть модель предсказывает v_0, а мы прибавляем его к z0.
        v_student = self.student(z0, t=torch.zeros(batch_size, device=device))
        x_fake = z0 + v_student  # Интегрирование Эйлера за 1 шаг (dt=1)

        # ==========================================
        # 2. REGRESSION LOSS (Trajectory Matching)
        # "Будь как учитель на конкретных примерах"
        # ==========================================
        # Чтобы посчитать честный регрессионный лосс, нам нужно знать,
        # куда бы пришел учитель из этого же шума z0.
        # Это дорого (требует запуска солвера), поэтому часто используют
        # технику "Pseudo-Huber" или сравнивают на промежуточных точках.
        # Для простоты здесь мы используем "Reflow" подход:

        with torch.no_grad():
            # Запускаем учителя на 1-2 шага или полный солвер (дорого, но точно)
            # В реальном DMD2 здесь часто используется аппроксимация.
            # Допустим, мы заранее закэшировали пары (z0, real_x) или используем real_images как target
            # Но для честного FM-Reflow нужно сгенерировать таргет учителем:
            target_x = self.run_teacher_solver(z0)

        loss_reg = F.mse_loss(x_fake, target_x)

        # ==========================================
        # 3. DISTRIBUTION LOSS (Адаптация SDS для Flow)
        # "Генерируй реалистичные картинки, даже если они не точные копии"
        # ==========================================

        # Берем случайное время t для проверки качества (не 0 и не 1, а между)
        t_vals = torch.rand(batch_size, device=device)

        # Зашумляем фейковую картинку до времени t
        # В Flow Matching (OT) интерполяция линейная: x_t = (1-t)*x_0 + t*x_1
        # Но здесь мы идем ОБРАТНО от сгенерированного x_fake к шуму?
        # Нет, мы просто интерполируем между новым шумом и x_fake.
        noise_new = torch.randn_like(x_fake)
        x_fake_t = (1 - t_vals[:, None, None, None]) * noise_new + t_vals[
            :, None, None, None
        ] * x_fake

        # Спрашиваем учителя: "Какая скорость (направление) здесь правильная?"
        with torch.no_grad():
            v_teacher = self.teacher(x_fake_t, t_vals)

        # --- КЛЮЧЕВОЙ МОМЕНТ АДАПТАЦИИ ---
        # В диффузии (SDS) мы считаем градиент как (pred_noise - target_noise).
        # В Flow Matching вектором направления является сама скорость v.
        # v_teacher указывает на "истинные данные" x_1.

        # Мы хотим, чтобы x_fake находился там, куда указывает v_teacher.
        # Предсказанная учителем точка данных x_1_pred:
        # x_1_pred = x_fake_t + (1 - t) * v_teacher

        x_target_from_teacher = x_fake_t + (1 - t_vals[:, None, None, None]) * v_teacher

        # Лосс распределения:
        # Мы хотим сдвинуть x_fake в сторону x_target_from_teacher.
        # Используем трюк с градиентами (как в SDS), чтобы не дифференцировать через учителя.

        grad_direction = (x_target_from_teacher - x_fake).detach()

        # Максимизируем сходство (или минимизируем расстояние)
        # loss = x_fake * grad_direction (для SDS стиля) или просто MSE
        # Более стабильный вариант - MSE с detach таргета:
        loss_dist = F.mse_loss(x_fake, x_target_from_teacher.detach())

        # ==========================================
        # ИТОГ
        # ==========================================
        # alpha балансирует между точностью соответствия шуму и реализмом
        total_loss = loss_reg + 0.5 * loss_dist
        return total_loss

    def run_teacher_solver(self, z0, steps=10):
        # Простой эйлеров солвер для учителя (для получения таргета регрессии)
        x = z0
        dt = 1.0 / steps
        for i in range(steps):
            t = i / steps
            t_tensor = torch.ones(z0.shape[0], device=z0.device) * t
            v = self.teacher(x, t_tensor)
            x = x + v * dt
        return x
