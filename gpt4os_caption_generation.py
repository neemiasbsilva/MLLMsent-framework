import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import warnings

from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from models.minigpt4.common.config import Config
from models.minigpt4.common.dist_utils import get_rank
from models.minigpt4.common.registry import registry
from models.minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from models.minigpt4.datasets.builders import *
from models.minigpt4.models import *
from models.minigpt4.processors import *
from models.minigpt4.runners import *
from models.minigpt4.tasks import *
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('-p', '--data_path')
    parser.add_argument('-b', '--num_beams')
    parser.add_argument('-t', '--temperature')
    parser.add_argument('-s', '--save_path')
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



def image_to_tensor(image_path, target_size=(224, 224)):
    """
    Loads an image from the specified path and converts it into a PyTorch tensor.

    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): A tuple specifying the target size for the image (default is (224, 224)).

    Returns:
    - tensor (torch.Tensor): A PyTorch tensor representing the image.
    """

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # Load image using Pillow (PIL)
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    tensor = preprocess(image)
    
    return tensor


def load_image(image_path, chat, chat_state=None):
    tensor = image_to_tensor(image_path)
    img_list = [tensor]
    chat.encode_img(img_list)
    return chat_state, img_list


def gradio_ask(user_message, chat, chat_state=None):
    chat.ask(user_message, chat_state)
    return '', chat_state


def gradio_answer(chat, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(
        conv=chat_state, 
        img_list=img_list, 
        num_beams=num_beams, 
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return llm_message


def save_captions(df, save_path, id_prompt):
    # df.to_csv(os.path.join(save_path, 'image_caption_temp_01.csv'), index=False)
    # df.to_csv(os.path.join(save_path, 'image_caption_missing_data.csv'), index=False)
    # df.to_csv(os.path.join(save_path, 'image_caption_random_test.csv'), index=False)
    # df.to_csv(os.path.join(save_path, 'percept_dataset_alpha4_p5.csv'), index=False)
    # df.to_csv(os.path.join(save_path, 'percept_dataset_alpha5_p5.csv'), index=False)

    df.to_csv(os.path.join(save_path, f'prompt{id_prompt}_percept_dataset_alpha5_p3.csv'), index=False)


def main():
    warnings.filterwarnings("ignore")

    # ========================================
    #             Model Initialization
    # ========================================

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    data_path = args.data_path
    img_names = os.listdir(data_path)
    num_beams = int(args.num_beams)
    temperature = float(args.temperature)
    save_path = args.save_path

    df = pd.DataFrame({})
    df_percept = pd.read_csv("/home/neemias/perceptsent/notebooks/perceptions.csv")
    # only database that has labels
    # df_label = pd.read_csv("/home/neemias/flickr-kat/data/labels.csv") # For Flickr-kat
    # df_label = pd.read_csv("/home/neemias/image-sentiment/data/labels.csv") # For Image Sent
    # img_labels = df_label["path"].to_list()
    # img_labels = [img.split('/')[-1] for img in img_labels]

    ### Load experiments for alpha 3
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha3_p5.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha3_p3.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha3_p2plus.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha3_p2neg.csv")

    ### Load experiments for alpha 4
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha4_p5.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha4_p3.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha4_p2plus.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha4_p2neg.csv")

    ### Load experiments for alpha 5
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha5_p5.csv")
    df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha5_p3.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha5_p2plus.csv")
    # df_label = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/minigpt4-classify/percept_dataset_alpha5_p2neg.csv")
    df_messages = pd.read_csv("/home/neemias/PerceptSent-LLM-approach/data/prompts.csv")
    _, df_label = train_test_split(df_label, test_size=0.2, random_state=42)

    img_labels = df_label["id"].to_list()
    img_labels = [img+'.jpg' for img in img_labels]
    # print(img_labels[:10])
    # print(img_names[:10])
    img_names = [img for img in img_names if img in img_labels]
    messages = df_messages["prompt"].to_list()
    id_prompts = df_messages["id"].to_list()

    for message, id_prompt in zip(messages, id_prompts):
        for img_name in tqdm(img_names, desc="Image captioning progress"):

            model_config = cfg.model_cfg
            model_config.device_8bit = args.gpu_id
            model_cls = registry.get_model_class(model_config.arch)
            model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
            CONV_VISION = conv_dict[model_config.model_type]


            vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
            vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

            
            stop_words_ids = [[835], [2277, 29937]]
            stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

            chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
            image_path = os.path.join(data_path, img_name)
            chat_state, image_list = load_image(image_path=image_path, chat=chat, chat_state=CONV_VISION.copy())
            # id = image_path.split('/')[-1].split('.')[0]
            # if (id in df_percept["id"].to_list()):
            #     perceptions = str(df_percept[df_percept["id"] == id]["perceptions"].iloc[0])
            # Message to generate the description for zero-shot and finetuning
            # message = """
            #     <Img><ImageHere></Img> Describe this image in detail.
            # """

            # Message to generate the sentiment classification.
            # message = """
            # """
            _, chat_state = gradio_ask(user_message=f"{str(message)}.",# The perceptions of the image: {perceptions}.", 
                                        chat=chat, chat_state=CONV_VISION.copy())
            
            llm_message = gradio_answer(chat, chat_state, image_list, num_beams, temperature)
            df = pd.concat([df, pd.DataFrame({"image_path": [str(image_path)], "caption": [str(llm_message)]})], axis=0)
            # save_captions(df, save_path)
            save_captions(df, save_path, id_prompt)
            torch.cuda.empty_cache()
            del stopping_criteria
            del stop_words_ids
            del model_cls
            del model
            del chat

        # save_captions(df, save_path)
        save_captions(df, save_path, id_prompt)


if __name__ == "__main__":
    print('Initialization Finished')

    main()