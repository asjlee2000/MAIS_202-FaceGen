from keras.models import load_model
from matplotlib import pyplot
import numpy as np

def create_latent(dim_latent, num_samples):
    latent = []
    # Draw from random distribution
    for i in range(num_samples):
        latent.append(np.random.randn(dim_latent))
    # Convert latent list to nparray
    latent = np.array(latent)
    return latent

def generate(model):
    latent = create_latent(100, 16)

    result = model.predict(latent)
    result = (result + 1) / 2.0
    return result

model = load_model("generator_weights_200.h5")

for plot in range(10):

    result = generate(model)

    for i in range(16):
        pyplot.subplot(4, 4, i + 1)
        pyplot.axis('off')
        pyplot.imshow(result[i])

    fn = 'generated_fakes_%d.png' % (plot + 1)
    pyplot.savefig(fn)
    pyplot.close()