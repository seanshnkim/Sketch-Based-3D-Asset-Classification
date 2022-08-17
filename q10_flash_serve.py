

def serve():
    from flash.image import ImageClassifier

    model = ImageClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/0.7.0/image_classification_model.pt"
    )
    model.serve(output="labels")



def send():
    import base64
    from pathlib import Path

    import requests

    import flash

    with (Path(flash.ASSETS_ROOT) / "fish.jpg").open("rb") as f:
        imgstr = base64.b64encode(f.read()).decode("UTF-8")

    body = {"session": "UUID", "payload": {"inputs": {"data": imgstr}}}
    resp = requests.post("http://127.0.0.1:8000/predict", json=body)
    print(resp.json())





if __name__ == "__main__":
    serve()
    
    
    