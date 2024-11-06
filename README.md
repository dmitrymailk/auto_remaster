# AUTO GAME REMASTER

## 07.11.24 SDXL+ControlNet+Reshade canny shader -> Venhancer

### SDXL video + Venhancer

#### Venhancer
- [workflow.json](showcases/showcase_2/nfs_venhancer.json)
- [youtube](https://youtu.be/7mpGQl7Z91k?si=XTTGQO5ZtLi6yMqI)
- <video src="https://github.com/user-attachments/assets/d623456a-8a91-4e24-a02b-0069cbb44b76" width="50%" controls autoplay loop></video>

#### SDXL
- [youtube](https://youtu.be/tdmL3rGf3NE)
- <video src="https://github.com/user-attachments/assets/959ea41b-b7dc-4217-a1d1-5d26841d2e2b" width="50%" controls autoplay loop></video>

#### Original
- [youtube](https://youtu.be/wYxjkZ3cXA8)
- <video src="https://github.com/user-attachments/assets/325fe3fd-6deb-45a0-abf1-38de7ccdb733" width="50%" controls autoplay loop></video>

### SDXL+ControlNet+Reshade
- [workflow.json](showcases/showcase_3/nfs_reshade_ip_lora_control_video_notes.json)
- [dataset](https://huggingface.co/dim/auto_remaster/blob/main/reshade_video_4.zip)
- ![117](showcases/showcase_3/117.png)
- ![138](showcases/showcase_3/138.png)
- ![195](showcases/showcase_3/195.png)
- ![246](showcases/showcase_3/246.png)

Моя гипотеза о том что canny фильтр из шейдера стабилизирует картинку оказалась верна. Пришлось заменить некоторые control nets и добавить несколько лор, чтобы сделать картинку более интересной, но теперь объекты в далеке расплываются гораздо меньше. В основном от этого страдают машины, наверное если написать шейдер который будет делать больший акцент на них, то всё решится. Также в моем пайплайне нет лор заточенных на автомобили, что могло бы(возможно) сделать их очертания более адекватными.

Проблема записи видео совместно с шейдером остается актуальной. На данный момент я написал тупой [скрипт](comfyui_sandbox/reshade_screenshot_recorder.py), который очень часто делает скриншоты через Reshade, однако даже на моем самом мощном ПК это выдает 10-12 фпс в лучшем случае. Как повысить производительность? Я думаю это можно сделать, например, через форк от Reshade и дописать функцию для записи видео, [вот тут делается скрин](https://github.com/crosire/reshade/blob/a0027100fe1eb360dcbb096f36b0a5edfa88aea1/source/runtime.cpp#L4804).

Однако это не решит того что мне не нужно применять шейдер для игрока, только взять результат вычисления.   

## 05.11.24 Venhancer, Mochi

#### Venhancer
- [workflow.json](showcases/showcase_2/nfs_venhancer.json)
- [dataset](https://huggingface.co/dim/auto_remaster/blob/main/render_nfs_noblur_high_graph_2_ip_control_lora_1.tar.gz)
- [youtube](https://youtu.be/KWcad7MjKQo)
- <video src="https://github.com/user-attachments/assets/66eb4ee4-d3bf-4190-9810-1e697165f8f2" width="50%" controls autoplay loop></video>


#### Mochi
- [workflow.json](showcases/showcase_2/nfs_mochi_enhancer.json)
- [dataset](https://huggingface.co/dim/auto_remaster/blob/main/render_nfs_noblur_high_graph_2_ip_control_lora_1.tar.gz)
- [youtube](https://youtu.be/h2xGpse_GRQ)
- <video src="https://github.com/user-attachments/assets/9633fbde-3ee5-4f51-8c66-9b032b33bb1a" width="50%" controls autoplay loop></video>

#### Original
- [youtube](https://youtu.be/UHwW8Y2Vyjs)
- [dataset](https://huggingface.co/dim/auto_remaster/blob/main/render_nfs_noblur_high_graph_2_ip_control_lora_1.tar.gz)
- <video src="https://github.com/user-attachments/assets/0454d387-4e55-4302-a9f1-4153786d03ab" width="50%" controls autoplay loop></video>

На мой взгляд [Venhancer](https://github.com/Vchitect/VEnhancer) справляется намного лучше чем mochi, хотя в обеих моделях я недостаточно экспериментировал над промптами и гиперпараметрами.

Картинка стала намного стабильнее, больше нет раздражающей ряби, однако потерялся изначальный стиль и вайб картинок. Однако я думаю это можно будет решить обычной лорой.Также мне кажется что на основе Venhancer сделать некий [refiner](https://www.reddit.com/r/StableDiffusion/comments/15ah7uj/can_someone_explain_what_the_sdxl_refiner_does/), чтобы картинка стала более интересной. Нам мой взгляд теперь стоит сосредоточиться на стабилизации картинки для control net.

## 03.11.24 SDXL+ControlNet+IPadapter+loras stack
- [workflow.json](./showcases/showcase_1/nfs_ip_control_lora_showcase_1.json)
- [dataset](https://huggingface.co/dim/auto_remaster/blob/main/render_nfs_noblur_high_graph_2.tar.gz)
- ![183](showcases/showcase_1/nfs_00000183.png)
- ![306](showcases/showcase_1/nfs_00000306.png)
- ![402](showcases/showcase_1/nfs_00000402.png)
- ![902](showcases/showcase_1/nfs_00000902.png)
- ![1279](showcases/showcase_1/nfs_00001279.png)
### videos
- [![video youtube](./showcases/showcase_1/showcase_video_preview.jpg)](https://youtu.be/AX1ZpzI6wcQ)
- [video youtube, style only](https://youtu.be/fegY0VjZm1A)
- [video github](showcases/showcase_1/output.mp4)

#### Преимущества
- хорошая работа со светом
- хорошая детализация многих текстур
- очень натуральные деревья
#### Недостатки
- машины и объекты в далеке появляются и исчезают
- картинка в видео слишком скачет
- произвольное мигание цветов

#### Возможное решение
Так как картинки для controlnet производятся на основе картинки из игры, модели которые совершают это делают серьезные ошибки, так же у них нет никакой информации об объекте, которые вот вот появится на экране. Поэтому я предлагают решить это при помощи шейдера. Один из примеров лежит тут, но он у меня не работает с nfs по дефолту в reshade(https://reshade.me/forum/shader-presentation/4635-mesh-edges). Я написал какой-то супербазовый скрипт для этого, но теперь встала главная проблема. **Как одновременно получать оригинальную картинку игры и картинку примененного шейдера?** *Я не знаю*.
- Скрипт лежит тут, [comfyui_sandbox/reshade_mesh_edges.fx](comfyui_sandbox/reshade_mesh_edges.fx)
![](showcases/showcase_1/original.jpg)
![](showcases/showcase_1/reshade.jpg)


### Useful Links
- [Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)
- [ComfyUI nodes for VEnhancer](https://github.com/kijai/ComfyUI-VEnhancer)
