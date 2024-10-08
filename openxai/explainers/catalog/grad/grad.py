import torch
from captum.attr import Saliency as Gradient_Captum
from ...api import BaseExplainer


class Gradient(BaseExplainer):
    """
    A baseline approach for computing input attribution.
    It returns the gradients with respect to inputs.
    https://arxiv.org/pdf/1312.6034.pdf
    """

    def __init__(self, model, absolute_value: bool = False) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        self.abs = absolute_value

        super(Gradient, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None) -> torch:
        """
        Explain an instance prediction.
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
            abs (bool): Returns absolute value of gradients if set to True, otherwise returns the (signed) gradients if False
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        self.model.eval()
        print(f"{self.model.eval()}", "---")
        self.model.zero_grad()
        label = self.model(x.float()).argmax(dim=-1) if label is None else label

        saliency = Gradient_Captum(self.model)

        attribution = saliency.attribute(x.float(), target=label, abs=self.abs)

        return attribution
