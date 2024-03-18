import sys
import torch
import transformers
sys.path.append("../..")
from otter.modeling_otter import OtterForConditionalGeneration
from pipeline.demo.util_mvtec import *

model_name = "context_true"
trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

model = OtterForConditionalGeneration.from_pretrained("/home/ueno/Otter/weights/OTTER-Image-MPT7B/", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
if train_ckpt.get("model_state_dict", None) is not None:
    train_ckpt = train_ckpt["model_state_dict"]
_ = model.load_state_dict(train_ckpt, strict=False)

folder = "wood"
sub_folder = ["color"]
# sub_folder = ["color","scratch","hole"]
GTs = ["stained wood","scratched wood","wood with holes"]
test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["liquid","scratch","hole"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)


folder = "bottle"
sub_folder = ["broken_large"]
# sub_folder = ["broken_large","contamination"]
GTs = ['broken bottle','contaminated bottle']
test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["broken_small","contamination"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","missing_cable","poke_insulation"]
# GTs = ['bent','swapp','crack','missing','hole']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["bent_wire","cable_swap","cut_outer_insulation","missing_wire","poke_insulation"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination"]
# GTs = ['stain','cut','hole','contamination']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["color","cut","hole","thread"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "grid"
# sub_folder = ["bent","broken","glue"]
# GTs = ["bent","broken","contamination"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["bent","broken","metal_contamination"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["bent","broken","thread"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "leather"
# sub_folder = ["color","cut","fold","poke"]
# GTs = ['stain','scratch','wrinkle','hole']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["glue","cut","fold","poke"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["glue","poke","fold","cut"]
# GTs = ['stain','hole','wrinkle','scratch']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch"]
# GTs = ['stain','contamination','crack','misprint','scratch']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["pill_type","contamination","crack","faulty_imprint","scratch"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "screw"
# sub_folder = ["scratch_head","manipulated_front"]
# GTs = ['chip','strip']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["scratch_neck","manipulated_front"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["thread_side","manipulated_front"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["thread_top","manipulatsub_foldered_front"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke"]
# GTs = ['crack','contamination','stain']
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["crack","rough","oil"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "wood"
# sub_folder = ["color","scratch","hole"]
# GTs = ["stained wood","scratched wood","wood with holes"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["liquid","scratch","hole"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","split_teeth"]
# GTs = ["broken","tear","frayed","misshapen"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)
# sub_folder = ["broken_teeth","fabric_border","rough","squeezed_teeth"]
# test_context(folder, sub_folder, GTs, model, model_name, image_processor)