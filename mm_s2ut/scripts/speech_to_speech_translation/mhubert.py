from itertools import groupby
import joblib, torch, torchaudio
import numpy
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class HubertCode(torch.nn.Module):
    def __init__(
        self, 
        hubert_model: str, 
        km_path: str, 
        km_layer: int = 11, 
        sampling_rate: int = 16000, 
        chunk_sec: int = 5
    ):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.sampling_rate = sampling_rate
        self.chunk_length = sampling_rate * chunk_sec
        self.km_model = joblib.load(km_path)
        self.km_layer = km_layer
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)

    def forward(self, source, attention_mask=None):
        output = self.model(
            input_values=source, 
            attention_mask=attention_mask, 
            output_attentions=True, 
            output_hidden_states=True, 
            return_dict=True, 
        )
        feature = output.hidden_states[self.km_layer]
        return {
            "feature": feature, 
            "padding_mask": output.attentions, 
        }
    
    def to(self, device):
        self.model.to(device)
        self.C = self.C.to(device)
        self.Cnorm = self.Cnorm.to(device)
        return self

    def decode(self, feature, beamsearch=True, top_k=10, beamsize=200):
        dist = torch.sqrt(
            feature.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feature, self.C)
            + self.Cnorm
        )
        min_dist = torch.topk(dist.detach(), top_k, dim=-1, largest=False)
        pred_ind_array = min_dist.indices.cpu().numpy()
        pred_values_array = min_dist.values.cpu().numpy()
        code_output = min_dist.indices.T.cpu().numpy()[0]

        return_dict = {
            'code': code_output,
            'distance': dist.detach().cpu().numpy(),
            'center_diff': (
                feature.cpu() - torch.index_select(
                    torch.tensor(
                        self.C_np.transpose()
                    ).cpu(), 0, min_dist.indices[:, 0].cpu())
            ).detach().numpy(),
            'merged_code': [k for k, _ in groupby(code_output)]
        }
        if beamsearch:
            sequences = [[[], 1.0]]
            for i_row, v_row in zip(pred_ind_array, pred_values_array):
                all_candidates = list()
                for seq in sequences:
                    tokens, score = seq
                    for k, v in zip(i_row, v_row):
                        norm_len_rate = (len([k for k, _ in groupby(tokens + [k])]) / len(code_output))
                        norm_dist_rate = (v / numpy.sum(v_row))
                        candidate = [tokens + [k], score + norm_len_rate * norm_dist_rate]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=False)
                sequences = ordered[:beamsize]
            code_output = sequences[0][0]
            return_dict['beam_code'] = code_output
            return_dict['beam_merged_code'] = [k for k, _ in groupby(code_output)]
        return return_dict