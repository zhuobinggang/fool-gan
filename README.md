# fool-gan(Make GAN easily!)

## DCGAN

```
from dcgan import *
m = DCGAN()
train_vanilla(m, epoch=60) # Will auto output 60 images to current folder
G.plot_random(m)
```

## Result(DCGAN)

[animation.md](Click me to show animation)

## InfoGAN

```
from infogan_dc_based import *
import runner as R
m = InfoGAN()
R.train_and_output_infogan(m, 100)
```
