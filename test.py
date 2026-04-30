# modify on top of https://github.com/xavihart/Diff-PGD

from PIL import ImageFilter
from load_dm import get_imagenet_dm_conf
from dataset import get_dataset
from utils import *
import torch
import torchvision
from tqdm.auto import tqdm
import random
from archs import get_archs, IMAGENET_MODEL
from advertorch.attacks import LinfPGDAttack, L2PGDAttack, SpatialTransformAttack
import matplotlib.pylab as plt
import time
import glob
import pandas as pd
from attack_tools import gen_pgd_confs
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import Resize, Grayscale, GaussianBlur
from torchmetrics.image import SSIM, PSNR
from torcheval.metrics import FrechetInceptionDistance as FID
from diffusers import LCMScheduler, TCDScheduler, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import *
from peft import LoraConfig
from datasets import load_dataset
import argparse
from autoattack import AutoAttack
import cv2
import lpips
from safetensors.torch import load_file
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader
from dame_recon.purifier import *
from advex_uar.attacks.snow_attack import *
from advex_uar.attacks.fog_attack import *
from advex_uar.attacks.gabor_attack import *

def parse_args():
    parser = argparse.ArgumentParser(description="choose using LCM/TCD and original/adversarial Lora")
    parser.add_argument("--model", required=True, type=str, help="tcd or lcm")
    parser.add_argument("--load_origin_lora", default=False, action="store_true",
                        help="using original/adversarial Lora")
    parser.add_argument("--lora_input_dir", type=str, help="input lora directory")
    parser.add_argument("--output_dir", default="vis_and_stat/", type=str, help="output directory")
    parser.add_argument("--num_validation_set", default=1000, type=int, help="size of subset of validation set")
    parser.add_argument("--num_inference_step", default=1, type=int, help="inference step of diffusion model")
    parser.add_argument("--strength", default=0.1, type=float, help="noise added to the original image")
    parser.add_argument("--seed", default=3407, type=int, help="seed for random number generator")
    parser.add_argument("--guidance_scale", default=1.0, type=float, help="guidance scale of diffusion model")
    parser.add_argument("--control_scale", default=0.8, type=float, help="control sclae of diffusion model")
    parser.add_argument("--input_image", default="./image_net", type=str, help="input image directory")
    parser.add_argument("--classifier", default="resnet50", type=str, help="classification model")
    parser.add_argument("--attack_method", default="Linf_pgd", type=str, help="attack model")
    parser.add_argument("--device", default=0, help="gpu:?")
    args = parser.parse_args()
    return args

def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict

class Denoised_Classifier(torch.nn.Module):
    def __init__(self, classifier, pipe, device, diffusion, model, t):
        super().__init__()
        self.classifier = classifier
        self.pipe = pipe
        self.device = device
        self.diffusion = diffusion
        self.model = model
        self.t = t
    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t

        x = x * 2 - 1

        t = torch.full((x.shape[0],), t).long().to(x.device)

        x_t = self.diffusion.q_sample(x, t)

        sample = x_t

        indices = list(range(t + 1))[::-1]

        # visualize
        l_sample = []
        l_predxstart = []

        for i in indices:
            # out = self.diffusion.ddim_sample(self.model, sample, t)
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0],), i).long().to(x.device))
   
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    def lcm_lora_denoise(self, x, pipe, device, to_512=False):
        batch = x.shape[0]
        prompt = ["" for _ in range(batch)]

        size = x.size()[-1]
        if size != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear")

        start=time.time()

        pil_x = TF.to_pil_image(x.squeeze(0))
        image = np.array(pil_x)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        control_image = control_image.resize((512, 512), resample=Image.NEAREST)

        x = torch.clamp(x, 0, 1)  # assume the input is 0-1
        generator = torch.manual_seed(args.seed)

        image = pipe(
            prompt=prompt,
            image=x,
            control_image=control_image,
            num_inference_steps=args.num_inference_step,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            controlnet_conditioning_scale=args.control_scale,
            generator=generator,
            output_type="pt",
            return_dict=False
        )

        out_image = F.interpolate(image[0], size=(size, size), mode="bilinear")

        end_time = time.time()
        print("run time:", end_time - start)

        return out_image, control_image.resize((size, size))

    def forward(self, x):
        out = self.lcm_lora_denoise(x, self.pipe, self.device)  # [0, 1]
        out = self.classifier(out)
        return out

def generate_x_adv(x, y, classifier, pgd_conf, device):
    net = classifier
    if args.attack_method == "Linf_pgd":
        adversary = LinfPGDAttack(net,
                                  loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                  eps=pgd_conf['eps'],
                                  nb_iter=pgd_conf['iter'],
                                  eps_iter=pgd_conf['alpha'],
                                  rand_init=False,
                                  targeted=False,
                                  )
        x_adv = adversary.perturb(x, y)

    elif args.attack_method == "L2_pgd":
        adversary = L2PGDAttack(net,
                                  loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                  eps=0.5,
                                  nb_iter=pgd_conf['iter'],
                                  eps_iter=0.1,
                                  rand_init=False,
                                  targeted=False,
                                  )
        x_adv = adversary.perturb(x, y)

    elif args.attack_method == "stadv":
        adversary = SpatialTransformAttack(net,
                                  num_classes=1000,
                                  clip_min=-1.,
                                  loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                  max_iterations=pgd_conf['iter'],
                                  )
        x_adv = adversary.perturb(x, y)

    elif args.attack_method == "AutoAttack":
        adversary = AutoAttack(classifier, norm='Linf', eps=pgd_conf["eps"], version='standard', device=device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=1)

    elif args.attack_method == "target_Linf_pgd":
        label_offset = torch.randint(low=1, high=1000, size=y.shape, generator=None).to(device)
        random_target = torch.remainder(y + label_offset, 1000).to(device)
        adversary = LinfPGDAttack(net,
                                  loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                  eps=pgd_conf['eps'],
                                  nb_iter=pgd_conf['iter'],
                                  eps_iter=pgd_conf['alpha'],
                                  rand_init=False,
                                  targeted=True,
                                  )
        x_adv = adversary.perturb(x, random_target)

    elif args.attack_method == "snow":
        x_adv = SnowAttack(
            nb_its=10, eps_max=0.0625, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    elif args.attack_method == "fog":
        x_adv = FogAttack(
            nb_its=10, eps_max=128, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    elif args.attack_method == "gabor":
        x_adv = GaborAttack(
            nb_its=10, eps_max=12.5, step_size=0.002236, resol=224
        )._forward(net, x*255, y, scale_eps=False, avoid_target=True)/255
    else:
        raise NotImplementedError
    return x_adv.to(device)


def generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t, pipe):
    net = Denoised_Classifier(classifier, pipe, device, diffusion, model, t)

    delta = torch.zeros(x.shape).to(x.device)
    # delta.requires_grad_()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    eps = pgd_conf['eps']
    alpha = pgd_conf['alpha']
    iter = pgd_conf['iter']

    for pgd_iter_id in range(iter):

        x_diff = net.sdedit(x+delta, t).detach()

        x_diff.requires_grad_()

        with torch.enable_grad():
            loss = loss_fn(classifier(x_diff), y)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)
    print("Done")

    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()

def Global(classifier, device, respace, t, args, eps=16, iter=10, name='attack_global', alpha=2, version="v1"):
    pgd_conf = gen_pgd_confs(eps=eps, alpha=alpha, iter=iter, input_range=(0, 1))
    if args.load_origin_lora:
        save_path = os.path.join(args.output_dir,
                                 f"{args.attack_method}/{args.classifier}/{args.model}/origin_lora_1/{args.lora_input_dir}/num_inference_step_{args.num_inference_step}_strength_{int(args.strength * 1000)}_guidance_scale_{args.guidance_scale}_{args.num_validation_set}_control_scale_{args.control_scale}")
    else:
        save_path = os.path.join(args.output_dir,
                                 f"{args.attack_method}/{args.classifier}/{args.model}/{args.lora_input_dir}/num_inference_step_{args.num_inference_step}_strength_{int(args.strength * 1000)}_guidance_scale_{args.guidance_scale}_{args.num_validation_set}_control_scale_{args.control_scale}")

    mp(save_path + "/visualization/")
    seed_everything(args.seed)
    classifier = get_archs(classifier, 'imagenet')  
    classifier = classifier.to(device)
    classifier.eval()

    dataset = get_dataset(
        'imagenet', split='test', adv=False
    )

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    model, diffusion = get_imagenet_dm_conf(device=device, respace=respace)

    c = 0

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
    ).to(device, dtype=torch.float32)

    # set scheduler
    if args.model == "LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif args.model == "TCD":
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError("model must be LCM")

    # load LoRA layer and it's weight
    if args.load_origin_lora:
        if args.model == "LCM":
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="origin_lora")
            if args.lora_input_dir != None:
                lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
                unwrapped_state_dict = {}

                for peft_key, weight in lora_state_dict[0].items():
                    key = peft_key.replace("base_model.model.", "")
                    unwrapped_state_dict[key] = weight.to(pipe.dtype)

                pipe.load_lora_weights(unwrapped_state_dict, adapter_name="adv_lora")
                pipe.set_adapters(["origin_lora", "adv_lora"], adapter_weights=[0.5, 0.5])
        elif args.model == "TCD":
            pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="origin_lora")
        else:
            raise ValueError("invalid model")
    else:
        lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
        unwrapped_state_dict = {}

        for peft_key, weight in lora_state_dict[0].items():
            key = peft_key.replace("base_model.model.", "")
            unwrapped_state_dict[key] = weight.to(pipe.dtype)

        pipe.load_lora_weights(unwrapped_state_dict)

    classifier_accuracy = 0
    attack_fail_rate = 0
    robust_accuracy = 0
    clean_accuracy = 0

    clean_typical_accuracy = 0
    typical_accuracy = 0

    mp(save_path + "/clean_image/")
    mp(save_path + "/robust_image/")
    mp(save_path + "/canny_image/")
    mp(save_path + "/adversarial_image/")
    mp(save_path + "/typical_image/")
    
    i = 1

    for x, y in test_loader:
        if i > args.num_validation_set:
            break

        x, y = x.to(device), y.to(device)

        classifier_accuracy += (y == classifier(x).argmax(1)).sum().item()

        if version == 'v1':
            x_adv = generate_x_adv(x, y, classifier, pgd_conf, device)            #pgd
        elif version == 'v2':
            x_adv = generate_x_adv_denoised_v2(x, y, diffusion, model, classifier, pgd_conf, device, t, pipe)     #diff-pgd

        attack_fail_rate += (y == classifier(x_adv).argmax(1)).sum().item()

        with (torch.no_grad()):
            net = Denoised_Classifier(classifier, pipe, device, diffusion, model, t)

            denoised_clean_x, natual_canny = net.lcm_lora_denoise(x, pipe, device)
            robust_x, adv_canny = net.lcm_lora_denoise(x_adv, pipe, device)
        
        si(x, save_path + f'/clean_image/{i}.png')
        si(robust_x, save_path + f'/robust_image/{i}.png')
        si(x_adv, save_path + f'/adversarial_image/{i}.png')
        
        clean_accuracy += (y == classifier(denoised_clean_x.to(torch.float32)).argmax(1)).sum().item()
        robust_accuracy += (y == classifier(robust_x.to(torch.float32)).argmax(1)).sum().item()
        i += 1

    stat = {
        "classifier_accuracy": classifier_accuracy / args.num_validation_set,
        "attack_fail_rate": attack_fail_rate/ args.num_validation_set,
        "clean_accuracy": clean_accuracy / args.num_validation_set,
        "robust_accuracy": robust_accuracy / args.num_validation_set,
        "clean_typical_accuracy": clean_typical_accuracy / args.num_validation_set,
        "typical_accuracy": typical_accuracy / args.num_validation_set,
    }

    stat = pd.DataFrame(stat, index=[0])
    stat.to_csv(save_path + f'stat.csv')

    print(stat)

if __name__ == '__main__':
    args = parse_args()
    Global(args.classifier, args.device, 'ddim50', t=150, eps=4, iter=1, name='attack_global_gradpass', alpha=1,                     #4/255 pgd-100 if want to run autoattack 4/255 just run this and add args.attack_method=AutoAttack
                args=args, version="v1")
