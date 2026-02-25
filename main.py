from model import MTT

if __name__ == "__main__":
    model = MTT()
    model.paramsCalc()
    model.load(checkpoint_path = "mtt_checkpoint.pt")
    results = model.translate(
        srcText=["Hello, how are you? The stock market crashed yesterday."],
        targetLang="vi",
        maxNewTokens=100,
        numBeams=4,
        device="cpu"
    )
    print(results)
