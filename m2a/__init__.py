"""mii2attr: a multi-task CNN that maps a rendered Mii image back to the discrete
Mii (Ver3 / Studio) attributes that produced it.

Pipeline
--------
1. ``schema``      - canonical attribute field ranges + head / nuisance split.
2. ``studio``      - sample random Miis and (de)serialise Mii Studio data.
3. ``renderer``    - turn Studio data into an image via an FFL render server.
4. ``dataset_gen`` - generate (image, label) pairs to disk + a manifest.
5. ``data``        - torch Dataset / transforms.
6. ``model``       - timm backbone + parallel per-attribute classification heads.
7. ``losses``      - summed cross-entropy + per-head accuracy.
8. ``train`` / ``eval`` / ``infer`` - training, evaluation, inference + re-render.
"""

__version__ = "0.1.0"
