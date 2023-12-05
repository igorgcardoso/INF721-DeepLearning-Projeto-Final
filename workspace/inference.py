from workspace.model import Supressor


def inference():
    model = Supressor.from_pretrained()
    # model = model.cuda()  # TODO: Uncomment this line if you want to use GPU
    model.eval()

    file = 'dataset/data/val/input/noisy3_SNRdb_20.0_clnsp3.wav'  # TODO: Replace this with the path to your audio file

    output = model.suppress(file)
    # TODO: Do something with the output


if __name__ == "__main__":
    inference()
