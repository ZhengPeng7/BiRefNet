import torch
from utils import check_state_dict
from models.birefnet import BiRefNet
from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter

weights_file = 'BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth'
device = ['cuda', 'cpu'][1]

def convert_to_onnx(net, file_name='output.onnx', input_shape=(1024, 1024), device=device):
    input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    input_layer_names = ['input_image']
    output_layer_names = ['output_image']

    torch.onnx.export(
        net,
        input,
        file_name,
        verbose=False,
        opset_version=17,
        input_names=input_layer_names,
        output_names=output_layer_names,
    )

with open('config.py') as fp:
    file_lines = fp.read()
if 'swin_v1_tiny' in weights_file:
    print('Set `swin_v1_tiny` as the backbone.')
    file_lines = file_lines.replace(
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][6]
        ''',
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][3]
        ''',
    )
    with open('config.py', mode="w") as fp:
        fp.write(file_lines)
else:
    file_lines = file_lines.replace(
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][3]
        ''',
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][6]
        ''',
    )
    with open('config.py', mode="w") as fp:
        fp.write(file_lines)
        
        



birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load(weights_file, map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
_ = birefnet.eval()


# register deform_conv2d operator
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()



convert_to_onnx(birefnet, weights_file.replace('.pth', '.onnx'), input_shape=(1024, 1024), device=device)









