# from google import genai
import json
import torch
from PIL import Image
import numpy as np
from ast import literal_eval

# Thresholds
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.4

OBJ_LIST = [
    "Apple",
    "Banana",
    "Brick",
    "Biscuit_Box",
    "Hammer",
    "Clamp",
    "Cup",
    "Mug",
    "Mustard_Bottle",
    "Pear",
    "Power_Drill",
    "Screwdriver",
    "Scissor",
    "Strawberry",
    "Tennis_Ball",
    "Soup_Can",
    "Strawberry",
]

# def post_process_resp(response):
#     text = response.text.strip()

#     # Remove square brackets if present
#     if text.startswith("[") and text.endswith("]"):
#         text = text[1:-1]

#     # Split by comma and strip spaces
#     objects = [x.strip() for x in text.split(",")]

#     return objects

def post_process_resp(response):
    return literal_eval(response.text)

def make_promt_return_all_object(object_list: list[str]):

    obj_string = "[" + ", ".join(object_list) + "]"

    prompt = (
        f"You are given a predefined set of valid object names: {obj_string}.\n\n"
        "Your task:\n"
        "1. Look at the provided [image].\n"
        "2. Identify ALL objects that appear in the image **from the predefined list only**.\n"
        "3. Return them as a Python list of strings in the exact format:\n"
        "   ['object1', 'object2', ...]\n"
        "4. If none of the objects appear in the image, return [].\n\n"
        "Rules:\n"
        "- Only include objects that are visible in the image.\n"
        "- Do NOT include objects that are not in the predefined list.\n"
        "- Do NOT guess ambiguous objects.\n"
        "- Do NOT output explanations or reasoning.\n"
        "- Output must be EXACTLY a Python list and nothing else.\n"
    )

    return prompt

def make_prompt_from_user_input(obj_list: list[str], user_input: str):

    obj_string = "[" + ", ".join(obj_list) + "]"

    prompt = (
        f"You are given a predefined set of valid object names: {obj_string}.\n"
        f"The user command is: \"{user_input}\".\n\n"
        "Your task:\n"
        "1. Look at objects from the predefined list.\n"
        "2. Find object(s) presenting in the given IMAGE are **most suitable** for fulfilling the user's command.\n"
        "3. Return a Python list of the selected object names in the format:\n"
        "   ['object1', 'object2']\n"
        "4. If no object is suitable for the command, return [].\n\n"
        "Rules:\n"
        "- Do NOT include objects that are not in the predefined list.\n"
        "- Do NOT output explanations or extra text.\n"
        "- Output must be ONLY a Python list.\n"
    )

    return prompt

def find_objects_with_prompt(img_path, prompt):
    client = genai.Client(api_key="AIzaSyCBg7qKWpl1b77AMwFaLCzF5hKh2thY4Cg")
    upload_file = client.files.upload(file=img_path)

    # Gemini Vision models require contents to be structured as a list of "parts"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"file_data": {"file_uri": upload_file.uri}}
                ]
            }
        ]
    )

    return post_process_resp(response)

def objects_detection_vlm(gd_processor, gd_model, sam_processor, sam_model, base_image, object_list):
    """
    Detect bbox of all objects
    """
    all_detections = []
    gd_device = gd_model.device
    sam_device = sam_model.device

    for i, obj_name in enumerate(object_list):
        
        prompt_name = obj_name.replace("_", " ").lower()
        text_prompt = f"A {prompt_name}."
        # --- Grounding DINO Step (Data moves to DINO_DEVICE) ---
        # Input data moves to DINO_DEVICE ('cuda')
        inputs = gd_processor(images=base_image, text=text_prompt, return_tensors="pt").to(gd_device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = gd_model(**inputs) # DINO inference on DINO_DEVICE

        # Post-process (DINO output is on DINO_DEVICE, target_sizes needs to match)
        target_sizes = torch.tensor([base_image.size[::-1]]).to(gd_device)
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=target_sizes
        )
        result = results[0]

        # --- Detection Filtering and Selection ---
        if len(result["scores"]) > 0:
            score = result["scores"][0].item()
            # Box coordinates MUST be moved to CPU for subsequent tensor creation
            box = result["boxes"][0].cpu().tolist() 
            label = result["text_labels"][0] 
            
            # ... (logging and metadata storage)
            xmin, ymin, xmax, ymax = box
            # print(f" → Found {label} (conf={score:.3f}) bbox={xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f}")
            detect_result = {
                "object": obj_name, 
                "confidence": round(score, 3), 
                "center": (round((xmin + xmax) / 2, 1), round((ymin + ymax) / 2, 1)),
                "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
            }
            
            # --- SAM Segmentation Step --- #
            try:
                # 1️⃣ Prepare input box on the SAM_DEVICE ('cpu')
                input_box = torch.tensor([box]).unsqueeze(0).to(sam_device)

                # 2️⃣ Process image and boxes
                sam_inputs = sam_processor(base_image, input_boxes=input_box, return_tensors="pt")
                sam_inputs = {k: v.to(sam_device) for k, v in sam_inputs.items()}
                
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)

                # 3️⃣ Post-process SAM to get the mask
                masks = sam_processor.post_process_masks(
                    sam_outputs.pred_masks,
                    sam_inputs["original_sizes"],
                    sam_inputs["reshaped_input_sizes"]
                )[0]

                # --- ⚠️ Robust Fix Starts Here ⚠️ ---
                # 4️⃣ Handle mask dimensions safely
                # (Do NOT call squeeze() before inspecting shape)

                # Usually: masks shape = (1, 1, H, W)
                if masks.ndim == 4:
                    mask_tensor = masks[0, 0]  # remove both batch + channel dims
                elif masks.ndim == 3:
                    mask_tensor = masks[0]  # remove only batch dim
                else:
                    mask_tensor = masks  # already 2D
                
                # Convert to numpy 2D array
                mask_np = mask_tensor.cpu().numpy()

                # Convert to uint8 (0–255)
                mask_data_for_pil = (mask_np * 255).astype(np.uint8)

                # 5️⃣ Create PIL mask and resize to image size
                mask_img = Image.fromarray(mask_data_for_pil)
                mask_img = mask_img.resize(base_image.size, Image.Resampling.NEAREST)

                # Optional overlay visualization
                mask_np = np.array(mask_img)
                detect_result["mask"] = mask_np
            except:
                print("SAM cannot run successfully")
                detect_result["mask"] = np.zeros(shape=base_image.size, dtype=np.uint8)

            # Append the detection result
            all_detections.append(detect_result)

        else:
            print(f" → No detection found for {obj_name}. Skipping segmentation.")

    return all_detections


if __name__ == "__main__":
    user_input = "I want to play a sport"
    img_path = f"save_images/color_0.png"
    prompt = make_prompt_from_user_input(user_input)
    output = find_objects_with_prompt(img_path, prompt)
    print(output)