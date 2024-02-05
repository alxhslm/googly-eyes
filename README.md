# :eyes: Googly eyes

The objective of this project is to be able to:
1. Detect faces in an image
2. Replace all of the eyes with googly eyes

The [RetinaFace](hhttps://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf) model is used to detect faces and eyes.

We then overlay Googly eyes with a random pupil size and orientation using [Pillow](https://pillow.readthedocs.io/en/stable/).

## Usage
1. Install Docker on your machine
2. Download the pre-trained model by running the following:
    ```
    wget https://github.com/alxhslm/googly-eyes/releases/download/v1.0/retinaface.tflite -O retinaface/retinaface.tflite
    ```
3. Run the following:
    ```
    docker compose up dashboard
    ```
    which will build and launch the Flask server and Streamlit dashboard, each in a separate container.
4. View the Streamlit dashboard in your browser by navigating to http://localhost:8501/.
