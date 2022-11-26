# Neural Font Rendering

This is the repository containing the code for the CVPR submission Neural Font Rendering.

Repo structure:

* `architecture experiments` contains the various architectures we experimented with:
    * `memorization_masked_mlp` is the masked MLP approach from the paper
    * `implicit` is the implicit representation approach from the paper
    * `implicit_multiple_weights` is the multi-weight experiment with weight interpolation (baesd on the implicit representation approach)
* `ground_truth_generation` contains code for generating ground truth images from `.ttf` files
* `visualizations` contains code for creating visualizations and cascade displays of results