from utils import get_model_complexity_info
import models_cifar as models
import torchvision.models

model = models.__dict__['prm_mnasnet1_0'](num_classes=1000)

flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
print('Flops:  %.3f' % (flops / 1e9))
print('Params: %.2fM' % (params / 1e6))
# print(model)
print(flops)
print(params)

# model = models.__dict__['se_mnasnet1_0'](num_classes=1000)
# print(model)
