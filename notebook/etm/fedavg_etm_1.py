import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F

print(torch.cuda.is_available())

import pandas as pd

from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import ranksums

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_topics", type=int, help="num_topics")

# Parse arguments
args = parser.parse_args()
if args.num_topics is None:
    parser.error("--num_topics is required")

# Set the seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

t_hidden_size = 512
rho_size = 512
num_topics = args.num_topics
enc_drop = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

icd_data = pd.read_csv("../../data/eicu_mimic_patient_diagnosis.csv")

print(f"Experiment: FedAvg ETM with {num_topics} topics, LR: 1e-4")

hospital_ids = [2001, 1001]

readmit_interval_threshold = {
    2001: 180, # Whether MIMIC patients will readmit to hospital within 180 days
    1001: 2 # Whether eICU patients will readmit to ICU within 2 days
}

train_loaders = {}
train_icds = {}
test_icds = {}
x_bow_tests = {}

train_readmit_row_ids = {}
test_readmit_row_ids = {}

train_label_deaths = {}
test_label_deaths = {}

train_label_readmit = {}
test_label_readmit = {}

for hospital_id in hospital_ids:

    hospital_data = icd_data[icd_data["hospitalid"] == hospital_id]
    train_data, test_data = train_test_split(hospital_data, test_size=0.2, random_state=42)

    x_train = train_data.iloc[:, 4:].to_numpy()
    x_test = test_data.iloc[:, 4:].to_numpy()

    print("Hospital ID:", hospital_id)
    print("Train data shape:", x_train.shape)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(x_train_tensor, x_train_tensor)  # Use the same tensor for inputs and targets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_loaders[hospital_id] = train_loader

    train_icds[hospital_id] = x_train_tensor
    test_icds[hospital_id] = x_test_tensor

    # Bag of words
    x_bow_test = []
    for row in x_test:
        word_id = list(np.where(row == 1)[0])
        x_bow_test.append(word_id)
    x_bow_tests[hospital_id] = x_bow_test

    # Readmission patients row ids
    train_data_np = train_data.to_numpy()
    train_readmit_patients_row_ids = np.where(train_data_np[:, 3] == 1)[0]
    train_readmit_row_ids[hospital_id] = train_readmit_patients_row_ids

    test_data_np = test_data.to_numpy()
    test_readmit_patients_row_ids = np.where(test_data_np[:, 3] == 1)[0]
    test_readmit_row_ids[hospital_id] = test_readmit_patients_row_ids

    # Label death in readmission
    y_death_train = train_data.iloc[train_readmit_patients_row_ids, 2].to_numpy()
    y_death_test = test_data.iloc[test_readmit_patients_row_ids, 2].to_numpy()
    train_label_deaths[hospital_id] = y_death_train
    test_label_deaths[hospital_id] = y_death_test


icd_code_names = icd_data.columns[4:]

train_data_total = icd_data
x_train = train_data_total.iloc[:, 4:].to_numpy()
x_bow_train = []
for row in x_train:
    word_id = list(np.where(row == 1)[0])
    x_bow_train.append(word_id)

# Train Word2Vec embeddings
# word2vec_model = Word2Vec(sentences=x_bow_train, vector_size=rho_size, window=5, min_count=1, sg=1)
# pretrained_rho = word2vec_model.wv.vectors
# pretrained_rho_tensor = torch.tensor(pretrained_rho, dtype=torch.float32)

def convert_icd9_to_disease(icd_9):
    if pd.isna(icd_9):
        return "Others"
    primary_icd9 = icd_9.split(',')[0].strip()
    try:
        # Convert the input to a float to handle both numeric and decimal ICD-9 codes
        icd_9_float = float(primary_icd9)

        # Check the ICD-9 code against the known ranges
        if 1 <= icd_9_float <= 139.9:
            return "Infection"
        elif 140 <= icd_9_float <= 239.9:
            return "Neoplasms"
        elif 240 <= icd_9_float <= 279.9:
            return "Endocrine"
        elif 280 <= icd_9_float <= 289.9:
            return "Blood"
        elif 290 <= icd_9_float <= 319:
            return "Mental"
        elif 320 <= icd_9_float <= 389.9:
            return "Nervous"
        elif 390 <= icd_9_float <= 459.9:
            return "Circulatory"
        elif 460 <= icd_9_float <= 519.9:
            return "Respiratory"
        elif 520 <= icd_9_float <= 579.9:
            return "Digestive"
        elif 580 <= icd_9_float <= 629.9:
            return "Genitourinary"
        elif 630 <= icd_9_float <= 676.9:
            return "Pregnancy"
        elif 680 <= icd_9_float <= 709.9:
            return "Skin"
        elif 710 <= icd_9_float <= 739.9:
            return "Musculoskeletal"
        elif 740 <= icd_9_float <= 759.9:
            return "Congenital"
        elif 760 <= icd_9_float <= 799.9:
            return "Perinatal"
        elif 800 <= icd_9_float <= 1000:
            return "Poisoning"
        elif icd_9.startswith("V"):
            return "Others"
        else:
            return "Others"

    except ValueError:
        return "Others"

disease_color_map = {
    "Infection": "#005896",
    "Neoplasms": "#dc5f00",      # SteelBlue
    "Endocrine": "#008002",      # LimeGreen
    "Blood": "#b40005",          # Crimson
    "Mental": "#74499c",         # DarkViolet
    "Nervous": "#6c382e",        # Gold
    "Circulatory": "#ab3db3",    # OrangeRed
    "Respiratory": "#2e2e2e",    # DarkTurquoise
    "Digestive": "#9c9c00",      # DeepPink
    "Genitourinary": "#009eac",  # MediumSlateBlue
    "Pregnancy": "#abcc25",      # HotPink
    "Skin": "#f06e60",           # SaddleBrown
    "Musculoskeletal": "#3bd156",# DarkOliveGreen
    "Congenital": "#c7b228",     # BlueViolet
    "Perinatal": "#ff5c7c",      # IndianRed
    "Poisoning": "#1268fd",      # DarkOrange
    "Others": "#696969",         # DimGray
    "Unknown": "#808080"         # Gray
}

hospital_color_map = {
    1001: "#1268fd",
    2001: "#ff5c7c",
}

icd_code_dict = dict() # Key: disease category, Value: list of ICD codes
for icd_code in icd_code_names:
    disease = convert_icd9_to_disease(icd_code)
    if disease in icd_code_dict:
        icd_code_dict[disease].append(icd_code)
    else:
        icd_code_dict[disease] = [icd_code]

icd_code_dict

patient_icd_data = icd_data.iloc[:, 4:]

total_feature_sum_dict = {}
for feature in patient_icd_data.columns:

    feature_sum = patient_icd_data[feature].sum()
    feature_name = convert_icd9_to_disease(feature)
    print(f"{feature_name}: {feature_sum}")

    if feature_name in total_feature_sum_dict:
        total_feature_sum_dict[feature_name] += feature_sum
    else:
        total_feature_sum_dict[feature_name] = feature_sum

print(total_feature_sum_dict)

total_feature_sum_list = []
for feature in patient_icd_data.columns:

    feature_name = convert_icd9_to_disease(feature)
    feature_sum = total_feature_sum_dict[feature_name]
    total_feature_sum_list.append(feature_sum)

feature_sums_tensor = torch.tensor(total_feature_sum_list)
feature_sums_tensor

def find_common_icds(input, feature_sums_tensor):

    most_common_icd_names = []
    least_common_icd_names = []

    for row in input:

        active_indices = (row == 1).nonzero(as_tuple=True)[0].cpu().numpy()

        if len(active_indices) == 0:
            least_common_icd_names.append("Others")
        else:
            active_sums = feature_sums_tensor[active_indices]
            _, max_idx = torch.max(active_sums, dim=0)
            most_common_feature_idx = active_indices[max_idx]
            most_common_feature_icd = patient_icd_data.columns[most_common_feature_idx.item()]
            most_common_feature_name = convert_icd9_to_disease(most_common_feature_icd)

            most_common_icd_names.append(most_common_feature_name)

            _, min_idx = torch.min(active_sums, dim=0)
            least_common_feature_idx = active_indices[min_idx]
            least_common_feature_icd = patient_icd_data.columns[least_common_feature_idx.item()]
            least_common_feature_name = convert_icd9_to_disease(least_common_feature_icd)

            least_common_icd_names.append(least_common_feature_name)

    return most_common_icd_names, least_common_icd_names

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k, :].argsort()[-topk:][::-1]
        list_w[k, :] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    return TD

def get_topic_coherence(beta, data, topk):
    D = len(data)  ## number of docs...data is list of documents
    TC = []
    num_topics = len(beta)
    counter = 0
    for k in range(num_topics):
        top_10 = list(beta[k].argsort()[-topk:][::-1])
        TC_k = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k)
    TC = np.mean(TC) / counter
    TC = (TC + 1) / 2
    return TC

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj


def top_k_precision(y_true, y_probs, k=3):
    top_k_indices = np.argsort(y_probs)[-k:]
    true_positives_in_top_k = np.sum(y_true[top_k_indices])
    top_k_precision = true_positives_in_top_k / k
    return top_k_precision

def need_aggregate(key):
    if key.startswith("rho") or key.startswith("q_theta"):
        return True
    else:
        return False


def average_weights(client_model_states, client_data_sizes):
    """Returns the weighted average of the model states provided by each client.

    Args:
    client_model_states (list): List of model states (dictionaries) from each client.
    client_data_sizes (list): List of data sizes for each client, used as weights.

    Returns:
    dict: The averaged model state dictionary.
    """
    total_data_points = sum(client_data_sizes)

    avg_state = {}
    for key, value in client_model_states[0].items():
        if need_aggregate(key):
            avg_state[key] = torch.zeros_like(value)

    for i, client_state in enumerate(client_model_states):
        weight = client_data_sizes[i] / total_data_points
        for key in avg_state.keys():
            avg_state[key] += client_state[key] * weight

    return avg_state


def average_evaluation(evaluations, client_data_sizes):
    """Returns the weighted average of the evaluation metrics provided by each client.

    Args:
    evaluations (list): List of evaluation metrics from each client.
    client_data_sizes (list): List of data sizes for each client, used as weights.

    Returns:
    float: The averaged evaluation metric.
    """
    total_data_points = sum(client_data_sizes)

    avg_evaluation = 0.0

    for i, evaluation in enumerate(evaluations):
        weight = client_data_sizes[i] / total_data_points
        avg_evaluation += evaluation * weight

    return avg_evaluation


def load_weights(model, updated_state_dict):
    original_state = model.state_dict()

    current_state_dict = {}
    for key, value in original_state.items():
        if need_aggregate(key) and key in updated_state_dict:
            # If need_aggregate is True, use the value from state_dict
            current_state_dict[key] = updated_state_dict[key]
        else:
            # Otherwise, use the original model state
            current_state_dict[key] = original_state[key]

    model.load_state_dict(current_state_dict)

# # Based on https://arxiv.org/pdf/1706.00359
# def topic_diversity_regularizer(topic_matrix):
#     num_topics = topic_matrix.size(0)
#
#     total_angles = []
#     for i in range(num_topics):
#         for j in range(i + 1, num_topics):
#
#             similarity = F.cosine_similarity(topic_matrix[i], topic_matrix[j], dim=0)
#             similarity = torch.clamp(similarity, -1.0, 1.0)
#             angles = torch.arccos(similarity)
#
#             total_angles.append(angles)
#
#     mean_angles = torch.mean(torch.stack(total_angles))
#     variance = torch.var(torch.stack(total_angles))
#     diversity_penalty = variance - mean_angles
#     # diversity_penalty = eps * variance - lambda_ * mean_angles
#
#     return diversity_penalty

def topic_diversity_regularizer(topic_matrix, threshold=0.1):
    # num_topics = topic_matrix.size(0)
    # diversity_penalty = 0
    # for i in range(num_topics):
    #     for j in range(i + 1, num_topics):
    #         similarity = F.cosine_similarity(topic_matrix[i], topic_matrix[j], dim=0)
    #         diversity_penalty += torch.max(torch.tensor(0.0), similarity - threshold)
    # return diversity_penalty
    normalized_topic_matrix = F.normalize(topic_matrix, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(normalized_topic_matrix, normalized_topic_matrix.t())
    mask = torch.eye(cosine_sim_matrix.size(0), device=cosine_sim_matrix.device).bool()
    cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0)
    diversity_penalty = torch.sum(torch.relu(cosine_sim_matrix - threshold)) / 2
    return diversity_penalty

disease_labels = [convert_icd9_to_disease(x) for x in icd_code_names]
disease_labels

def federated_learning(client_models, client_optimizers, rounds=10, epochs=1, enc_drop=0.5, beta_=0.05, lambda_=0.1):

    elbo_hist = {}
    kld_hist = {}
    recon_hist = {}

    tc_hist = {}
    td_hist = {}
    tq_hist = {}

    for client_id in hospital_ids:

        elbo_hist[client_id] = []
        kld_hist[client_id] = []
        recon_hist[client_id] = []

        tc_hist[client_id] = []
        td_hist[client_id] = []
        tq_hist[client_id] = []

    for round in range(rounds):

        client_model_states = []
        client_data_sizes = []

        for client_id in hospital_ids:

            client_loader = train_loaders[client_id]
            client_model = client_models[client_id]
            client_optimizer = client_optimizers[client_id]

            client_recon_likelihood = 0.0
            client_kld = 0.0
            client_elbo = 0.0

            client_model.train()

            for local_epoch in range(epochs):  # Local training epochs (can increase as needed)

                epoch_recon_likelihood = 0.0
                epoch_kld = 0.0
                epoch_elbo = 0.0

                for batch_idx, (bows, normalized_bows) in enumerate(client_loader):
                    bows = bows.to(device)
                    normalized_bows = normalized_bows.to(device)

                    client_optimizer.zero_grad()
                    recon_loss, kld_theta = client_model(bows, normalized_bows)

                    # Increase topic diversity
                    # Based on paper: https://arxiv.org/pdf/1706.00359
                    diversity_penalty = topic_diversity_regularizer(client_model.get_beta())

                    # Beta-VAE KL Annealing to prevent posterior collapse
                    kl_weight = min(1.0, round * beta_)
                    elbo_term = recon_loss + kld_theta
                    loss = recon_loss + kl_weight * kld_theta + lambda_ * diversity_penalty
                    loss.backward()
                    client_optimizer.step()

                    epoch_recon_likelihood += -recon_loss.item()
                    epoch_kld += kld_theta.item()
                    epoch_elbo += -elbo_term.item()

                epoch_recon_likelihood /= len(client_loader)
                epoch_kld /= len(client_loader)
                epoch_elbo /= len(client_loader)

                client_recon_likelihood += epoch_recon_likelihood
                client_kld += epoch_kld
                client_elbo += epoch_elbo

            client_recon_likelihood /= epochs
            client_kld /= epochs
            client_elbo /= epochs

            client_model_states.append(client_model.state_dict())  # Save client model state
            client_data_sizes.append(len(client_loader.dataset))

            elbo_hist[client_id].append(client_elbo)
            kld_hist[client_id].append(client_kld)
            recon_hist[client_id].append(client_recon_likelihood)

        # Aggregate the parameters from each client
        avg_model_state = average_weights(client_model_states, client_data_sizes)

        for client_id in hospital_ids:

            # Load the averaged model back into the client model
            client_model = client_models[client_id]
            load_weights(client_model, avg_model_state)

            # Evaluate
            client_model.eval()

            beta = client_model.get_beta()
            beta = beta.data.cpu().numpy()

            x_bow_test = x_bow_tests[client_id]
            coherence = get_topic_coherence(beta, x_bow_test, 5)
            diversity = get_topic_diversity(beta, 5)
            quality = coherence * diversity

            tc_hist[client_id].append(coherence)
            td_hist[client_id].append(diversity)
            tq_hist[client_id].append(quality)

            print(f"Round {round + 1}/{rounds} - Client: {client_id} - ELBO: {elbo_hist[client_id][-1]:.4f} - Recon likelihood: {recon_hist[client_id][-1]:.4f} - KLD: {kld_hist[client_id][-1]:.4f}, TC: {coherence:.4f}, TD: {diversity:.4f}, TQ: {quality:.4f}")

    return client_models, elbo_hist, kld_hist, recon_hist, tc_hist, td_hist, tq_hist


class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)

        ## define the word embedding matrix \rho
        self.rho = nn.Linear(rho_size, vocab_size, bias=False)

        # with torch.no_grad():
        #     self.rho.weight = nn.Parameter(pretrained_rho_tensor.T)
        #     self.rho.weight.requires_grad = False

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        # print(vocab_size, " THE Vocabulary size is here ")
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size),
                nn.ReLU(),
            )

        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            # During inference time, there is no need for random sampling.
            # Instead, the model can use the mean directly, which is a point estimate of the latent variable
            # This avoids unnecessary randomness during inference or testing.
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()

        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        """
        This generate the description as a defintion over words

        Returns:
            [type]: [description]
        """
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        # logit = self.alphas(self.rho.weight.T)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows, is_train=True, d=1.0):
        """
        getting the topic poportion for the document passed in the normalixe bow or tf-idf"""
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        if not is_train:
            theta = F.softmax(z / d, dim=-1)
        return z, theta, kld_theta

    def decode(self, theta, beta):
        """compute the probability of topic given the document which is equal to theta^T ** B

        Args:
            theta ([type]): [description]
            beta ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = torch.mm(theta, beta)

        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        predictions = torch.log(results_without_zeros)
        return predictions

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            _, theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

client_models = {}
client_optimizers = {}

for client_id in hospital_ids:

    client_model = ETM(num_topics=num_topics,
                       vocab_size=len(icd_code_names),
                       t_hidden_size=t_hidden_size,
                       rho_size=rho_size,
                       enc_drop=enc_drop).to(device)
    client_models[client_id] = client_model

    optimizer_fn = torch.optim.Adam(client_model.parameters(), lr=1e-4, weight_decay=5e-6)
    client_optimizers[client_id] = optimizer_fn

client_models, elbo_hist, kld_hist, recon_hist, tc_hist, td_hist, tq_hist = federated_learning(client_models, client_optimizers, rounds=200, epochs=5, enc_drop=enc_drop, beta_=0.01, lambda_=0.2)

# Plot ELBO
plt.clf()
plt.title("ELBO of ETM FedAvg (eICU)")
plt.plot(elbo_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.savefig(f"../result/fedavg_etm_{num_topics}_elbo_eicu.png", bbox_inches='tight')
plt.close()

# Plot ELBO
plt.clf()
plt.title("ELBO of ETM FedAvg of (MIMIC-III)")
plt.plot(elbo_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.savefig(f"../result/fedavg_etm_{num_topics}_elbo_mimic.png", bbox_inches='tight')
plt.close()

# Plot Reconstruction Likelihood
plt.clf()
plt.title("Reconstruction Likelihood of ETM FedAvg (eICU)")
plt.plot(recon_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("E[logp(x|z)]")
plt.savefig(f"../result/fedavg_etm_{num_topics}_recon_likelihood_eicu.png", bbox_inches='tight')
plt.close()

# Plot Reconstruction Likelihood
plt.clf()
plt.title("Reconstruction Likelihood of ETM FedAvg (MIMIC-III)")
plt.plot(recon_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("E[logp(x|z)]")
plt.savefig(f"../result/fedavg_etm_{num_topics}_recon_likelihood_mimic.png", bbox_inches='tight')
plt.close()

# Plot KL
plt.clf()
plt.title("KL[q(z|x) || p(z)] of ETM FedAvg (eICU)")
plt.plot(kld_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("KL[q(z|x) || p(z)]")
plt.savefig(f"../result/fedavg_etm_{num_topics}_kl_eicu.png", bbox_inches='tight')
plt.close()

# Plot KL
plt.clf()
plt.title("KL[q(z|x) || p(z)] of ETM FedAvg (MIMIC-III)")
plt.plot(kld_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("KL[q(z|x) || p(z)]")
plt.savefig(f"../result/fedavg_etm_{num_topics}_kl_mimic.png", bbox_inches='tight')
plt.close()

# Plot Topic Coherence
plt.clf()
plt.title("Topic Coherence of ETM FedAvg (eICU)")
plt.plot(tc_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("Topic Coherence")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_coherence_eicu.png", bbox_inches='tight')
plt.close()

# Plot Topic Coherence
plt.clf()
plt.title("Topic Coherence of ETM FedAvg (MIMIC-III)")
plt.plot(tc_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("Topic Coherence")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_coherence_mimic.png", bbox_inches='tight')
plt.close()

# Plot Topic Diversity
plt.clf()
plt.title("Topic Diversity of ETM FedAvg (eICU)")
plt.plot(td_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("Topic Diversity")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_diversity_eicu.png", bbox_inches='tight')
plt.close()

# Plot Topic Diversity
plt.clf()
plt.title("Topic Diversity of ETM FedAvg (MIMIC-III)")
plt.plot(td_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("Topic Diversity")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_diversity_mimic.png", bbox_inches='tight')
plt.close()

# Plot Topic Quality
plt.clf()
plt.title("Topic Quality (Coherence x Diversity) of ETM FedAvg (eICU)")
plt.plot(tq_hist[1001])
plt.xlabel("Epoch")
plt.ylabel("Topic Quality")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_quality_eicu.png", bbox_inches='tight')
plt.close()

# Plot Topic Quality
plt.clf()
plt.title("Topic Quality (Coherence x Diversity) of ETM FedAvg (MIMIC-III)")
plt.plot(tq_hist[2001])
plt.xlabel("Epoch")
plt.ylabel("Topic Quality")
plt.savefig(f"../result/fedavg_etm_{num_topics}_topic_quality_mimic.png", bbox_inches='tight')
plt.close()

eicu_client_model = client_models[1001]
eicu_topic_word_distribution = eicu_client_model.get_beta()
eicu_topic_word_distribution = eicu_topic_word_distribution.data.cpu().numpy()

total_top_icd_idx = np.zeros((eicu_topic_word_distribution.shape[0], 5))  # K x 5

for topic in range(eicu_topic_word_distribution.shape[0]):
    topic_icds = eicu_topic_word_distribution[topic, :]
    top_icd_idx = np.flip(np.argsort(topic_icds))[:5]  # Top 5 ICD codes
    total_top_icd_idx[topic] = top_icd_idx

total_top_icd_idx = np.ravel(total_top_icd_idx).astype(int)

eicu_total_top_icd = eicu_topic_word_distribution[:, total_top_icd_idx]
eicu_total_top_icd = eicu_total_top_icd.T

total_top_icd_names = icd_code_names[total_top_icd_idx]
disease = [convert_icd9_to_disease(x) for x in total_top_icd_names]
disease_label = [f"{disease[i]} - {total_top_icd_names[i]}" for i in range(len(disease))]

plt.clf()
plt.figure(figsize=(8, 10))

# Plot heatmap
plt.title("Heatmap of the Top 5 ICD Codes per Topic using ETM FedAvg (eICU)")
ax = sns.heatmap(eicu_total_top_icd,
            yticklabels=disease_label,
            cmap='Reds', vmax=0.2)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

y_labels = plt.gca().get_yticklabels()
for i, label in enumerate(y_labels):
    color = disease_color_map[disease[i]]
    label.set_color(color)

plt.savefig(f"../result/fedavg_etm_{num_topics}_beta_heatmap_eicu.png", bbox_inches='tight')
plt.close()

mimic_client_model = client_models[2001]
mimic_topic_word_distribution = mimic_client_model.get_beta()
mimic_topic_word_distribution = mimic_topic_word_distribution.data.cpu().numpy()

total_top_icd_idx = np.zeros((mimic_topic_word_distribution.shape[0], 5))  # K x 5

for topic in range(mimic_topic_word_distribution.shape[0]):
    topic_icds = mimic_topic_word_distribution[topic, :]
    top_icd_idx = np.flip(np.argsort(topic_icds))[:5]  # Top 5 ICD codes
    total_top_icd_idx[topic] = top_icd_idx

total_top_icd_idx = np.ravel(total_top_icd_idx).astype(int)

mimic_total_top_icd = mimic_topic_word_distribution[:, total_top_icd_idx]
mimic_total_top_icd = mimic_total_top_icd.T

total_top_icd_names = icd_code_names[total_top_icd_idx]
disease = [convert_icd9_to_disease(x) for x in total_top_icd_names]
disease_label = [f"{disease[i]} - {total_top_icd_names[i]}" for i in range(len(disease))]

plt.clf()
plt.figure(figsize=(8, 10))

# Plot heatmap
plt.title("Heatmap of the Top 5 ICD Codes per Topic using ETM FedAvg (MIMIC-III)")
ax = sns.heatmap(mimic_total_top_icd,
            yticklabels=disease_label,
            cmap='Reds', vmax=0.2)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

y_labels = plt.gca().get_yticklabels()
for i, label in enumerate(y_labels):
    color = disease_color_map[disease[i]]
    label.set_color(color)

plt.savefig(f"../result/fedavg_etm_{num_topics}_beta_heatmap_mimic.png", bbox_inches='tight')
plt.close()

eicu_x_bow_test = x_bow_tests[1001]
eicu_coherence = get_topic_coherence(eicu_topic_word_distribution, eicu_x_bow_test, 3)
eicu_diversity = get_topic_diversity(eicu_topic_word_distribution, 3)
eicu_quality = eicu_coherence * eicu_diversity

print("eICU ETM FedAvg Topic Coherence: ", eicu_coherence)
print("eICU ETM FedAvg Topic Diversity: ", eicu_diversity)
print("eICU ETM FedAvg Topic Quality: ", eicu_quality)

mimic_x_bow_test = x_bow_tests[2001]
mimic_coherence = get_topic_coherence(mimic_topic_word_distribution, mimic_x_bow_test, 3)
mimic_diversity = get_topic_diversity(mimic_topic_word_distribution, 3)
mimic_quality = mimic_coherence * mimic_diversity

print("MIMIC ETM FedAvg Topic Coherence: ", mimic_coherence)
print("MIMIC ETM FedAvg Topic Diversity: ", mimic_diversity)
print("MIMIC ETM FedAvg Topic Quality: ", mimic_quality)

disease_labels = [convert_icd9_to_disease(x) for x in icd_code_names]

mimic_client_model = client_models[2001]
eicu_x_test_tensor = test_icds[1001]

_, eicu_test_theta_unweighted, _ = mimic_client_model.get_theta(eicu_x_test_tensor)
eicu_test_theta_unweighted = eicu_test_theta_unweighted.data.cpu().numpy()

eicu_test_readmit_row_ids = test_readmit_row_ids[1001]
X_eicu_test = eicu_test_theta_unweighted[eicu_test_readmit_row_ids]
eicu_icd_input = eicu_x_test_tensor[eicu_test_readmit_row_ids]

eicu_most_common_icd_names, eicu_least_common_icd_names = find_common_icds(eicu_icd_input, feature_sums_tensor)

unique_diseases = np.unique(eicu_least_common_icd_names)
row_colors = pd.Series(eicu_least_common_icd_names).map(disease_color_map).to_numpy()

# Create a seaborn clustermap
plt.clf()
row_clusters = linkage(X_eicu_test, method='ward')
col_clusters = linkage(X_eicu_test.T, method='ward')
g = sns.clustermap(X_eicu_test, row_linkage=row_clusters, col_linkage=col_clusters,
                   figsize=(5, 6),
                   yticklabels=False, cmap='rocket_r',
                   cbar_kws={'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.6},
                   cbar_pos=(0.45, -0.05, 0.3, 0.02),
                   row_colors=row_colors)

g.fig.suptitle(f'Heatmap FedAvg ETM Patient-Topic Mixture (MIMIC on eICU)',
               fontsize=12, x=0.6, y=1.02)
g.ax_heatmap.set_xlabel('Latent Dimension')
g.ax_heatmap.set_ylabel('Patients')

legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=disease_color_map[disease],
                             markersize=10, label=disease) for disease in unique_diseases]
plt.legend(handles=legend_patches, title='Disease', bbox_to_anchor=(2.0, 12), loc='lower left', borderaxespad=0.)

plt.savefig(f"../result/fedavg_etm_{num_topics}_theta_heatmap_mimic_on_eicu.png", bbox_inches='tight')
plt.close()

unique_diseases = icd_code_dict.keys()
eicu_disease_p_values = {}

eicu_disease_topic_p_values = np.zeros((len(unique_diseases), num_topics))

for disease_idx, disease in enumerate(unique_diseases):

    # ICD codes for disease category
    icd_codes = icd_code_dict[disease]
    icd_code_idx = [np.where(icd_code_names == icd_code)[0][0] for icd_code in icd_codes]
    eicu_icd_input_disease = eicu_icd_input[:, icd_code_idx]

    # Find the patients if the disease is present
    patient_has_disease_indices = (eicu_icd_input_disease.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    patient_no_disease_indices = (eicu_icd_input_disease.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    patient_has_disease_indices = patient_has_disease_indices.cpu().numpy()
    patient_no_disease_indices = patient_no_disease_indices.cpu().numpy()

    if len(patient_has_disease_indices) == 0 or len(patient_no_disease_indices) == 0:
        continue

    smallest_p_value = np.inf
    for topic in range(num_topics):
        theta_topic = X_eicu_test[:, topic]
        theta_topic_has_disease = theta_topic[patient_has_disease_indices]
        theta_topic_no_disease = theta_topic[patient_no_disease_indices]
        _, p_value = ranksums(theta_topic_has_disease, theta_topic_no_disease)
        eicu_disease_topic_p_values[disease_idx][topic] = -np.log10(p_value)
        if p_value < smallest_p_value:
            smallest_p_value = p_value

    eicu_disease_p_values[disease] = -np.log10(smallest_p_value)

preg_other_idx = [idx for idx in range(len(unique_diseases)) if list(unique_diseases)[idx] == "Pregnancy" or list(unique_diseases)[idx] == "Others"]

eicu_disease_topic_p_values = np.delete(eicu_disease_topic_p_values, preg_other_idx, axis=0)
unique_diseases = np.delete(list(unique_diseases), preg_other_idx)

plt.clf()
plt.figure(figsize=(5, 6))
ax = sns.heatmap(eicu_disease_topic_p_values, cmap='Reds', cbar=True, cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'shrink': 0.8})

ax.set_yticklabels(unique_diseases, rotation=0)

colorbar = ax.collections[0].colorbar
colorbar.set_label("-log10(p-value)")

plt.xlabel('Topics')
plt.title('FedAvg ETM Significance of Disease-Topic Associations (MIMIC on eICU)')
plt.savefig(f"../result/fedavg_etm_{num_topics}_disease_topic_p_values_mimic_on_eicu.png", bbox_inches='tight')
plt.close()

eicu_client_model = client_models[1001]
mimic_x_test_tensor = test_icds[2001]

_, mimic_test_theta_unweighted, _ = eicu_client_model.get_theta(mimic_x_test_tensor)
mimic_test_theta_unweighted = mimic_test_theta_unweighted.data.cpu().numpy()
mimic_test_readmit_row_ids = test_readmit_row_ids[2001]

X_mimic_test = mimic_test_theta_unweighted[mimic_test_readmit_row_ids]
mimic_icd_input = mimic_x_test_tensor[mimic_test_readmit_row_ids]

mimic_most_common_icd_names, mimic_least_common_icd_names = find_common_icds(mimic_icd_input, feature_sums_tensor)

unique_diseases = np.unique(mimic_least_common_icd_names)
row_colors = pd.Series(mimic_least_common_icd_names).map(disease_color_map).to_numpy()

# Create a seaborn clustermap
plt.clf()
row_clusters = linkage(X_mimic_test, method='ward')
col_clusters = linkage(X_mimic_test.T, method='ward')
g = sns.clustermap(X_mimic_test, row_linkage=row_clusters, col_linkage=col_clusters,
                   figsize=(5, 6),
                   yticklabels=False, cmap='rocket_r',
                   cbar_kws={'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.6},
                   cbar_pos=(0.45, -0.05, 0.3, 0.02),
                   row_colors=row_colors)

g.fig.suptitle(f'Heatmap FedAvg ETM Patient-Topic Mixture (eICU on MIMIC)',
               fontsize=12, x=0.6, y=1.02)
g.ax_heatmap.set_xlabel('Latent Dimension')
g.ax_heatmap.set_ylabel('Patients')

legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=disease_color_map[disease],
                             markersize=10, label=disease) for disease in unique_diseases]
plt.legend(handles=legend_patches, title='Disease', bbox_to_anchor=(2.0, 12), loc='lower left', borderaxespad=0.)

plt.savefig(f"../result/fedavg_etm_{num_topics}_theta_heatmap_eicu_on_mimic.png", bbox_inches='tight')
plt.close()

unique_diseases = icd_code_dict.keys()
mimic_disease_p_values = {}

mimic_disease_topic_p_values = np.zeros((len(unique_diseases), num_topics))

for disease_idx, disease in enumerate(unique_diseases):

    # ICD codes for disease category
    icd_codes = icd_code_dict[disease]
    icd_code_idx = [np.where(icd_code_names == icd_code)[0][0] for icd_code in icd_codes]
    mimic_icd_input_disease = mimic_icd_input[:, icd_code_idx]

    # Find the patients if the disease is present
    patient_has_disease_indices = (mimic_icd_input_disease.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    patient_no_disease_indices = (mimic_icd_input_disease.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    patient_has_disease_indices = patient_has_disease_indices.cpu().numpy()
    patient_no_disease_indices = patient_no_disease_indices.cpu().numpy()

    if len(patient_has_disease_indices) == 0 or len(patient_no_disease_indices) == 0:
        continue

    smallest_p_value = np.inf
    for topic in range(num_topics):
        theta_topic = X_mimic_test[:, topic]
        theta_topic_has_disease = theta_topic[patient_has_disease_indices]
        theta_topic_no_disease = theta_topic[patient_no_disease_indices]
        _, p_value = ranksums(theta_topic_has_disease, theta_topic_no_disease)
        mimic_disease_topic_p_values[disease_idx][topic] = -np.log10(p_value)
        if p_value < smallest_p_value:
            smallest_p_value = p_value

    mimic_disease_p_values[disease] = -np.log10(smallest_p_value)


preg_other_idx = [idx for idx in range(len(unique_diseases)) if list(unique_diseases)[idx] == "Pregnancy" or list(unique_diseases)[idx] == "Others"]

mimic_disease_topic_p_values = np.delete(mimic_disease_topic_p_values, preg_other_idx, axis=0)
unique_diseases = np.delete(list(unique_diseases), preg_other_idx)

plt.clf()
plt.figure(figsize=(5, 6))
ax = sns.heatmap(mimic_disease_topic_p_values, cmap='Reds', cbar=True, cbar_kws={'orientation': 'horizontal', 'pad': 0.12, 'shrink': 0.8})

ax.set_yticklabels(unique_diseases, rotation=0)

colorbar = ax.collections[0].colorbar
colorbar.set_label("-log10(p-value)")

plt.xlabel('Topics')
plt.title('FedAvg ETM Significance of Disease-Topic Associations (eICU on MIMIC)')
plt.savefig(f"../result/fedavg_etm_{num_topics}_disease_topic_p_values_eicu_on_mimic.png", bbox_inches='tight')
plt.close()

mimic_client_model = client_models[2001]

mimic_x_train_tensor = train_icds[2001]
eicu_x_test_tensor = test_icds[1001]

_, mimic_train_theta_unweighted, _ = mimic_client_model.get_theta(mimic_x_train_tensor)
_, eicu_test_theta_unweighted, _ = mimic_client_model.get_theta(eicu_x_test_tensor)

mimic_train_theta_unweighted = mimic_train_theta_unweighted.data.cpu().numpy()
eicu_test_theta_unweighted = eicu_test_theta_unweighted.data.cpu().numpy()

mimic_train_readmit_row_ids = train_readmit_row_ids[2001]
eicu_test_readmit_row_ids = test_readmit_row_ids[1001]

X_mimic_train = mimic_train_theta_unweighted[mimic_train_readmit_row_ids]
X_eicu_test = eicu_test_theta_unweighted[eicu_test_readmit_row_ids]

y_mimic_death_train = train_label_deaths[2001]
y_eicu_death_test = test_label_deaths[1001]

# log_reg = LogisticRegression()
# log_reg.fit(X_mimic_train, y_mimic_death_train)
#
# y_eicu_death_cross_lr_scores = log_reg.predict_proba(X_eicu_test)[:, 1]
#
# precision_eicu_death_cross_lr, recall_eicu_death_cross_lr, _ = precision_recall_curve(y_eicu_death_test, y_eicu_death_cross_lr_scores)
# auprc_eicu_death_cross_lr = average_precision_score(y_eicu_death_test, y_eicu_death_cross_lr_scores)
# top_k_precision_death_cross_lr = top_k_precision(y_eicu_death_test, y_eicu_death_cross_lr_scores, k=100)
#
# fpr_eicu_death_cross_lr, tpr_eicu_death_cross_lr, _ = roc_curve(y_eicu_death_test, y_eicu_death_cross_lr_scores)
# auroc_eicu_death_cross_lr = roc_auc_score(y_eicu_death_test, y_eicu_death_cross_lr_scores)
#
# # Print the AUPRC and AUROC values
# print(f"AUPRC of Mortality Prediction after Re-admission (MIMIC on eICU): {auprc_eicu_death_cross_lr:.4f}")
# print(f"AUROC of Mortality Prediction after Re-admission (MIMIC on eICU): {auroc_eicu_death_cross_lr:.4f}")
# print(f"Top k Precision of Mortality Prediction after Re-admission (MIMIC on eICU): {top_k_precision_death_cross_lr:.4f}")

knn = KNeighborsClassifier(n_neighbors=120)
knn.fit(X_mimic_train, y_mimic_death_train)

y_eicu_death_cross_knn_scores = knn.predict_proba(X_eicu_test)[:, 1]

precision_eicu_death_cross_knn, recall_eicu_death_cross_knn, _ = precision_recall_curve(y_eicu_death_test, y_eicu_death_cross_knn_scores)
auprc_eicu_death_cross_knn = average_precision_score(y_eicu_death_test, y_eicu_death_cross_knn_scores)
top_k_precision_death_cross_knn = top_k_precision(y_eicu_death_test, y_eicu_death_cross_knn_scores, k=100)

fpr_eicu_death_cross_knn, tpr_eicu_death_cross_knn, _ = roc_curve(y_eicu_death_test, y_eicu_death_cross_knn_scores)
auroc_eicu_death_cross_knn = roc_auc_score(y_eicu_death_test, y_eicu_death_cross_knn_scores)

# Print the AUPRC and AUROC values
print(f"AUPRC of Mortality Prediction after Re-admission (MIMIC on eICU): {auprc_eicu_death_cross_knn:.4f}")
print(f"AUROC of Mortality Prediction after Re-admission (MIMIC on eICU): {auroc_eicu_death_cross_knn:.4f}")
print(f"Top k Precision of Mortality Prediction after Re-admission (MIMIC on eICU): {top_k_precision_death_cross_knn:.4f}")


eicu_client_model = client_models[1001]

eicu_x_train_tensor = train_icds[1001]
mimic_x_test_tensor = test_icds[2001]

_, eicu_train_theta_unweighted, _ = eicu_client_model.get_theta(eicu_x_train_tensor)
_, mimic_test_theta_unweighted, _ = eicu_client_model.get_theta(mimic_x_test_tensor)

eicu_train_theta_unweighted = eicu_train_theta_unweighted.data.cpu().numpy()
mimic_test_theta_unweighted = mimic_test_theta_unweighted.data.cpu().numpy()

eicu_train_readmit_row_ids = train_readmit_row_ids[1001]
mimic_test_readmit_row_ids = test_readmit_row_ids[2001]

X_eicu_train = eicu_train_theta_unweighted[eicu_train_readmit_row_ids]
X_mimic_test = mimic_test_theta_unweighted[mimic_test_readmit_row_ids]

y_eicu_death_train = train_label_deaths[1001]
y_mimic_death_test = test_label_deaths[2001]

# log_reg = LogisticRegression()
# log_reg.fit(X_eicu_train, y_eicu_death_train)
#
# y_mimic_death_cross_lr_scores = log_reg.predict_proba(X_mimic_test)[:, 1]
#
# precision_mimic_death_cross_lr, recall_mimic_death_cross_lr, _ = precision_recall_curve(y_mimic_death_test, y_mimic_death_cross_lr_scores)
# auprc_mimic_death_cross_lr = average_precision_score(y_mimic_death_test, y_mimic_death_cross_lr_scores)
# top_k_precision_mimic_cross_lr = top_k_precision(y_mimic_death_test, y_mimic_death_cross_lr_scores, k=100)
#
# fpr_mimic_death_cross_lr, tpr_mimic_death_cross_lr, _ = roc_curve(y_mimic_death_test, y_mimic_death_cross_lr_scores)
# auroc_mimic_death_cross_lr = roc_auc_score(y_mimic_death_test, y_mimic_death_cross_lr_scores)
#
# # Print the AUPRC and AUROC values
# print(f"AUPRC of Mortality Prediction after Re-admission (eICU on MIMIC): {auprc_mimic_death_cross_lr:.4f}")
# print(f"AUROC of Mortality Prediction after Re-admission (eICU on MIMIC): {auroc_mimic_death_cross_lr:.4f}")
# print(f"Top k Precision of Mortality Prediction after Re-admission (eICU on MIMIC): {top_k_precision_mimic_cross_lr:.4f}")

knn = KNeighborsClassifier(n_neighbors=120)
knn.fit(X_eicu_train, y_eicu_death_train)

y_mimic_death_cross_knn_scores = knn.predict_proba(X_mimic_test)[:, 1]

precision_mimic_death_cross_knn, recall_mimic_death_cross_knn, _ = precision_recall_curve(y_mimic_death_test, y_mimic_death_cross_knn_scores)
auprc_mimic_death_cross_knn = average_precision_score(y_mimic_death_test, y_mimic_death_cross_knn_scores)
top_k_precision_mimic_cross_knn = top_k_precision(y_mimic_death_test, y_mimic_death_cross_knn_scores, k=100)

fpr_mimic_death_cross_knn, tpr_mimic_death_cross_knn, _ = roc_curve(y_mimic_death_test, y_mimic_death_cross_knn_scores)
auroc_mimic_death_cross_knn = roc_auc_score(y_mimic_death_test, y_mimic_death_cross_knn_scores)

# Print the AUPRC and AUROC values
print(f"AUPRC of Mortality Prediction after Re-admission (eICU on MIMIC): {auprc_mimic_death_cross_knn:.4f}")
print(f"AUROC of Mortality Prediction after Re-admission (eICU on MIMIC): {auroc_mimic_death_cross_knn:.4f}")
print(f"Top k Precision of Mortality Prediction after Re-admission (eICU on MIMIC): {top_k_precision_mimic_cross_knn:.4f}")

with open(f"../result/fedavg_etm_{num_topics}_recall_eicu_death_cross_knn_values.txt", "w") as f:
    for r in recall_eicu_death_cross_knn:
        f.write(f"{r}\n")

with open(f"../result/fedavg_etm_{num_topics}_precision_eicu_death_cross_knn_values.txt", "w") as f:
    for p in precision_eicu_death_cross_knn:
        f.write(f"{p}\n")

with open(f"../result/fedavg_etm_{num_topics}_auprc_eicu_death_cross_knn.txt", "w") as f:
    f.write(f"{auprc_eicu_death_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_top_k_precision_eicu_cross_knn.txt", "w") as f:
    f.write(f"{top_k_precision_death_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_fpr_eicu_death_cross_knn_values.txt", "w") as f:
    for fp in fpr_eicu_death_cross_knn:
        f.write(f"{fp}\n")

with open(f"../result/fedavg_etm_{num_topics}_tpr_eicu_death_cross_knn_values.txt", "w") as f:
    for tp in tpr_eicu_death_cross_knn:
        f.write(f"{tp}\n")

with open(f"../result/fedavg_etm_{num_topics}_auroc_eicu_death_cross_knn.txt", "w") as f:
    f.write(f"{auroc_eicu_death_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_eicu_disease_p_values.txt", 'w') as file:
    for key, value in eicu_disease_p_values.items():
        file.write(f'{key}: {value}\n')

with open(f"../result/fedavg_etm_{num_topics}_eicu_disease_topic_p_values.txt", 'w') as file:
    for row in eicu_disease_topic_p_values:
        file.write(" ".join(f"{value:.5f}" for value in row) + "\n")

with open(f"../result/fedavg_etm_{num_topics}_recall_mimic_death_cross_knn_values.txt", "w") as f:
    for r in recall_mimic_death_cross_knn:
        f.write(f"{r}\n")

with open(f"../result/fedavg_etm_{num_topics}_precision_mimic_death_cross_knn_values.txt", "w") as f:
    for p in precision_mimic_death_cross_knn:
        f.write(f"{p}\n")

with open(f"../result/fedavg_etm_{num_topics}_auprc_mimic_death_cross_knn.txt", "w") as f:
    f.write(f"{auprc_mimic_death_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_top_k_precision_mimic_cross_knn.txt", "w") as f:
    f.write(f"{top_k_precision_mimic_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_mimic_disease_p_values.txt", 'w') as file:
    for key, value in mimic_disease_p_values.items():
        file.write(f'{key}: {value}\n')

with open(f"../result/fedavg_etm_{num_topics}_fpr_mimic_death_cross_knn_values.txt", "w") as f:
    for fp in fpr_mimic_death_cross_knn:
        f.write(f"{fp}\n")

with open(f"../result/fedavg_etm_{num_topics}_tpr_mimic_death_cross_knn_values.txt", "w") as f:
    for tp in tpr_mimic_death_cross_knn:
        f.write(f"{tp}\n")

with open(f"../result/fedavg_etm_{num_topics}_auroc_mimic_death_cross_knn.txt", "w") as f:
    f.write(f"{auroc_mimic_death_cross_knn}\n")

with open(f"../result/fedavg_etm_{num_topics}_mimic_disease_topic_p_values.txt", 'w') as file:
    for row in mimic_disease_topic_p_values:
        file.write(" ".join(f"{value:.5f}" for value in row) + "\n")

with open(f"../result/fedavg_etm_{num_topics}_unique_disease_names.txt", 'w') as file:
    for disease in unique_diseases:
        file.write(f'{disease}\n')

torch.save(client_models[2001].state_dict(), f'../result/fedavg_etm_{num_topics}mimic_client_model.pth')
torch.save(client_models[1001].state_dict(), f'../result/fedavg_etm_{num_topics}eicu_client_model.pth')

mimic_train_icds = train_icds[2001].cpu().numpy()
np.save(f"../result/fedavg_etm_{num_topics}mimic_train_icds.npy", mimic_train_icds)

mimic_test_icds = test_icds[2001].cpu().numpy()
np.save(f"../result/fedavg_etm_{num_topics}mimic_test_icds.npy", mimic_test_icds)

mimic_train_readmit_row_ids = train_readmit_row_ids[2001]
np.save(f"../result/fedavg_etm_{num_topics}mimic_train_readmit_row_ids.npy", mimic_train_readmit_row_ids)

mimic_test_readmit_row_ids = test_readmit_row_ids[2001]
np.save(f"../result/fedavg_etm_{num_topics}mimic_test_readmit_row_ids.npy", mimic_test_readmit_row_ids)

mimic_train_label_deaths = train_label_deaths[2001]
np.save(f"../result/fedavg_etm_{num_topics}mimic_train_label_deaths.npy", mimic_train_label_deaths)

mimic_test_label_deaths = test_label_deaths[2001]
np.save(f"../result/fedavg_etm_{num_topics}mimic_test_label_deaths.npy", mimic_test_label_deaths)

eicu_train_icds = train_icds[1001].cpu().numpy()
np.save(f"../result/fedavg_etm_{num_topics}eicu_train_icds.npy", eicu_train_icds)

eicu_test_icds = test_icds[1001].cpu().numpy()
np.save(f"../result/fedavg_etm_{num_topics}eicu_test_icds.npy", eicu_test_icds)

eicu_train_readmit_row_ids = train_readmit_row_ids[1001]
np.save(f"../result/fedavg_etm_{num_topics}eicu_train_readmit_row_ids.npy", eicu_train_readmit_row_ids)

eicu_test_readmit_row_ids = test_readmit_row_ids[1001]
np.save(f"../result/fedavg_etm_{num_topics}eicu_test_readmit_row_ids.npy", eicu_test_readmit_row_ids)

eicu_train_label_deaths = train_label_deaths[1001]
np.save(f"../result/fedavg_etm_{num_topics}eicu_train_label_deaths.npy", eicu_train_label_deaths)

eicu_test_label_deaths = test_label_deaths[1001]
np.save(f"../result/fedavg_etm_{num_topics}eicu_test_label_deaths.npy", eicu_test_label_deaths)