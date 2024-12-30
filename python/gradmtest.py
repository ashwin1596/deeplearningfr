import deeplearning as dl

config = dl.Config.get_instance()

config.set_device_type('GPU')
config.set_cuda_devices('0')


print("Is gradient mode enabled?", dl.GradMode.is_enabled())

with dl.no_grad():
	print("Is gradient mode enabled?", dl.GradMode.is_enabled())

print("Is gradient mode enabled?", dl.GradMode.is_enabled())

with dl.grad():
    print("Is gradient mode enabled?", dl.GradMode.is_enabled())