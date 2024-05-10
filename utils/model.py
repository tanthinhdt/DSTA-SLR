import torch
import numpy as np
import onnxruntime as ort


def get_predictions(
    inputs: np.ndarray,
    ort_session: ort.InferenceSession,
    id2gloss: dict,
    k: int = 3,
) -> list:
    '''
    Get the top-k predictions.

    Parameters
    ----------
    inputs : dict
        Model inputs.
    ort_session : ort.InferenceSession
        ONNX Runtime session.
    id2gloss : dict
        Mapping from class index to class label.
    k : int, optional
        Number of predictions to return, by default 3.

    Returns
    -------
    list
        Top-k predictions.
    '''
    if inputs is None:
        return []

    logits = torch.from_numpy(ort_session.run(None, {'x': inputs})[0])

    # Get top-3 predictions
    topk_scores, topk_indices = torch.topk(logits, k, dim=1)
    topk_scores = torch.nn.functional.softmax(topk_scores, dim=1).squeeze().detach().numpy()
    topk_indices = topk_indices.squeeze().detach().numpy()

    return [
        {
            'label': id2gloss[topk_indices[i]],
            'score': topk_scores[i],
        }
        for i in range(k)
    ]
