import torch
import torch.nn.functional as F
import numpy as np
import lpips
import piqa
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict
from PIL import Image


class ImageEvaluator:
    class EvaluationDataset(Dataset):
        def __init__(
            self,
            originals: List[Image.Image],
            generated: List[Image.Image],
            metrics_list: List[str],
            resolution: int,
        ):
            self.originals = originals
            self.generated = generated
            self.metrics_list = metrics_list
            self.resolution = resolution

            self.common_transform = transforms.Compose(
                [
                    transforms.Resize(
                        resolution, interpolation=transforms.InterpolationMode.LANCZOS
                    ),
                    transforms.CenterCrop(resolution),
                ]
            )
            self.to_tensor = transforms.ToTensor()
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        def __len__(self):
            return len(self.originals)

        def _process_image(self, pil_img):
            output = {}
            img_resized = self.common_transform(pil_img.convert("RGB"))

            need_base = any(
                m in self.metrics_list for m in ["ssim", "dists", "psnr", "mse"]
            )
            need_lpips = "lpips" in self.metrics_list
            need_fid = "fid" in self.metrics_list

            if need_base or need_lpips:
                tensor_base = self.to_tensor(img_resized)
                if need_base:
                    output["base"] = tensor_base
                if need_lpips:
                    output["lpips"] = self.normalize(tensor_base)

            if need_fid:
                arr = np.array(img_resized)
                tensor_fid = (
                    torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.uint8)
                )
                output["fid"] = tensor_fid

            return output

        def __getitem__(self, idx):
            return {
                "orig": self._process_image(self.originals[idx]),
                "gen": self._process_image(self.generated[idx]),
            }

    def __init__(
        self,
        metrics_list: List[str],
        device: str = "cuda",
        resolution: int = 512,
        num_workers: int = 4,
        prefix_key: str = "eval",
    ):
        self.metrics_list = [m.lower() for m in metrics_list]
        self.device = device
        self.resolution = resolution
        self.num_workers = num_workers
        self.prefix_key = prefix_key
        self._init_models()

    def _init_models(self):
        # 1. LPIPS (нет параметра reduction в init, делаем sum при вызове)
        if "lpips" in self.metrics_list:
            print("Initializing LPIPS...")
            self.loss_fn_lpips = (
                lpips.LPIPS(net="vgg").requires_grad_(False).to(self.device)
            )

        # 2. SSIM (piqa поддерживает reduction='sum')
        if "ssim" in self.metrics_list:
            print("Initializing SSIM...")
            self.loss_fn_ssim = piqa.SSIM(reduction="sum").to(self.device)

        # 3. TorchMetrics (PSNR, DISTS, FID) - они сами умеют накапливать состояние через update()
        if "dists" in self.metrics_list:
            print("Initializing DISTS...")
            self.metric_dists = DeepImageStructureAndTextureSimilarity().to(self.device)

        if "psnr" in self.metrics_list:
            print("Initializing PSNR...")
            self.metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if "fid" in self.metrics_list:
            print("Initializing FID...")
            self.metric_fid = FrechetInceptionDistance(feature=2048).to(self.device)

    def evaluate(
        self,
        originals: List[Image.Image],
        generated: List[Image.Image],
        batch_size: int = 16,
    ) -> Dict[str, float]:

        n_samples = len(originals)
        dataset = self.EvaluationDataset(
            originals, generated, self.metrics_list, self.resolution
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Сброс метрик torchmetrics
        if "dists" in self.metrics_list:
            self.metric_dists.reset()
        if "psnr" in self.metrics_list:
            self.metric_psnr.reset()
        if "fid" in self.metrics_list:
            self.metric_fid.reset()

        # Аккумуляторы для метрик, которые считаем вручную (LPIPS, MSE, SSIM)
        manual_sums = {
            k: 0.0 for k in ["lpips", "mse", "ssim"] if k in self.metrics_list
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):

                # --- LPIPS ---
                if "lpips" in self.metrics_list:
                    orig_l = batch["orig"]["lpips"].to(self.device, non_blocking=True)
                    gen_l = batch["gen"]["lpips"].to(self.device, non_blocking=True)
                    # LPIPS возвращает (B, 1, 1, 1), суммируем
                    manual_sums["lpips"] += (
                        self.loss_fn_lpips(orig_l, gen_l).sum().item()
                    )

                # --- Base Transforms (SSIM, MSE, DISTS, PSNR) ---
                need_base = any(
                    m in self.metrics_list for m in ["ssim", "dists", "psnr", "mse"]
                )
                if need_base:
                    orig_b = batch["orig"]["base"].to(self.device, non_blocking=True)
                    gen_b = batch["gen"]["base"].to(self.device, non_blocking=True)

                    # SSIM (piqa c reduction='sum')
                    if "ssim" in self.metrics_list:
                        manual_sums["ssim"] += self.loss_fn_ssim(orig_b, gen_b).item()

                    # MSE (ручной расчет суммы средних ошибок)
                    if "mse" in self.metrics_list:
                        # (gen - orig)^2 -> mean по пикселям -> sum по батчу
                        # Это дает нам сумму MSE каждой картинки
                        batch_mse_sum = (
                            F.mse_loss(gen_b, orig_b, reduction="none")
                            .mean(dim=[1, 2, 3])
                            .sum()
                            .item()
                        )
                        manual_sums["mse"] += batch_mse_sum

                    # DISTS (torchmetrics update)
                    if "dists" in self.metrics_list:
                        self.metric_dists.update(gen_b, orig_b)

                    # PSNR (torchmetrics update)
                    if "psnr" in self.metrics_list:
                        self.metric_psnr.update(gen_b, orig_b)

                # --- FID (torchmetrics update) ---
                if "fid" in self.metrics_list:
                    orig_f = batch["orig"]["fid"].to(self.device, non_blocking=True)
                    gen_f = batch["gen"]["fid"].to(self.device, non_blocking=True)
                    self.metric_fid.update(orig_f, real=True)
                    self.metric_fid.update(gen_f, real=False)

        # --- Сборка результатов ---
        final_metrics = {}

        # 1. Метрики с ручным суммированием делим на кол-во сэмплов
        for k, v in manual_sums.items():
            final_metrics[k] = v / n_samples

        # 2. Метрики torchmetrics вычисляют результат сами
        if "dists" in self.metrics_list:
            final_metrics["dists"] = float(self.metric_dists.compute().item())

        if "psnr" in self.metrics_list:
            final_metrics["psnr"] = float(self.metric_psnr.compute().item())

        if "fid" in self.metrics_list:
            print("Computing FID score...")
            final_metrics["fid"] = float(self.metric_fid.compute().item())

        # Округление
        for k, v in final_metrics.items():
            final_metrics[k] = round(v, 4)

        renamed_metrics = {}
        for k, v in final_metrics.items():
            renamed_metrics[f"{self.prefix_key}_{k}"] = v

        return renamed_metrics
