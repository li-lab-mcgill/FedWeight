class SimulateDataCreateService:

    def __init__(self, classifier):
        self._classifier = classifier

    def generate_fake_data(self, x_output):

        print("*********************")
        print(x_output.shape)
        print(self._classifier.classifier.weight)
        print("*********************")

        probs_output = self._classifier(x_output)

        probs_output = probs_output[:, 0]
        y_output = (probs_output >= 0.5).float()

        probs_output = probs_output.cpu().detach().numpy().copy()
        x_output = x_output.cpu().detach().numpy().copy()
        y_output = y_output.cpu().detach().numpy().copy()

        return probs_output, x_output, y_output
