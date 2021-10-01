from utils import *

PGM_RADIOLOGISTS = '''
data {
    int<lower=0> n_readers;
    int<lower=0> n_severities;
    int<lower=0> n_subgroups;
    int<lower=0> n_exams;
    int<lower=0> n_sides;
    real mu_mean;
    real gamma_mean;
    real nu_mean;
    real b_mean;
    real<lower=0> mu_sd;
    real<lower=0> gamma_sd;
    real<lower=0> nu_sd;
    real<lower=0> b_sd;
    int subgroups[n_exams, n_sides];
    int<lower=0, upper=1> pred[n_readers, n_severities, n_exams, n_sides];
    int<lower=0, upper=1> mask[n_readers, n_severities, n_exams];
}
parameters {
    real mu[n_exams, n_sides];
    real gamma_[n_severities-1, n_subgroups];
    real nu[n_readers, n_subgroups];
    real b[n_subgroups];
}
transformed parameters {
    real gamma[n_severities, n_subgroups];
    gamma[1, :] = rep_array(0.0, n_subgroups);
    for (severity_idx in 1:n_severities-1) {
        for (subgroup_idx in 1:n_subgroups) {
            gamma[severity_idx+1, subgroup_idx] = sum(gamma_[:severity_idx, subgroup_idx]);
        }
    }
}
model {
    // Specify priors
    for (exam_idx in 1:n_exams) {
        mu[exam_idx, 1] ~ normal(mu_mean, mu_sd);
        mu[exam_idx, 2] ~ normal(mu_mean, mu_sd);
    }
    for (subgroup_idx in 1:n_subgroups) {
        for (severity_idx in 1:n_severities-1) {
            gamma_[severity_idx, subgroup_idx] ~ normal(gamma_mean, gamma_sd);
        }
        for (reader_idx in 1:n_readers) {
            nu[reader_idx, subgroup_idx] ~ normal(nu_mean, nu_sd);
        }
        b[subgroup_idx] ~ normal(b_mean, b_sd);
    }

    // Specify model
    for (reader_idx in 1:n_readers) {
        for (severity_idx in 1:n_severities) {
            for (exam_idx in 1:n_exams) {
                if (mask[reader_idx, severity_idx, exam_idx] == 1) {
                    int subgroup_l = subgroups[exam_idx, 1];
                    int subgroup_r = subgroups[exam_idx, 2];
                    real theta_l = inv_logit(mu[exam_idx, 1] + gamma[severity_idx, subgroup_l] + nu[reader_idx, subgroup_l] + b[subgroup_l]);
                    real theta_r = inv_logit(mu[exam_idx, 2] + gamma[severity_idx, subgroup_r] + nu[reader_idx, subgroup_r] + b[subgroup_r]);
                    int pred_l = pred[reader_idx, severity_idx, exam_idx, 1];
                    int pred_r = pred[reader_idx, severity_idx, exam_idx, 2];
                    pred_l ~ bernoulli(theta_l);
                    pred_r ~ bernoulli(theta_r);
                }
            }
        }
    }
}
'''

PGM_DNNS = '''
data {
    int<lower=0> n_readers;
    int<lower=0> n_severities;
    int<lower=0> n_subgroups;
    int<lower=0> n_exams;
    int<lower=0> n_sides;
    real mu_mean;
    real gamma_mean;
    real nu_mean;
    real b_mean;
    real<lower=0> mu_sd;
    real<lower=0> gamma_sd;
    real<lower=0> nu_sd;
    real<lower=0> b_sd;
    int subgroups[n_exams, n_sides];
    real<lower=0, upper=1> pred[n_readers, n_severities, n_exams, n_sides];
    int<lower=0, upper=1> mask[n_readers, n_severities, n_exams];
}
parameters {
    real mu[n_exams, n_sides];
    real gamma_[n_severities-1, n_subgroups];
    real nu[n_readers, n_subgroups];
    real b[n_subgroups];
}
transformed parameters {
    real gamma[n_severities, n_subgroups];
    gamma[1, :] = rep_array(0.0, n_subgroups);
    for (severity_idx in 1:n_severities-1) {
        for (subgroup_idx in 1:n_subgroups) {
            gamma[severity_idx+1, subgroup_idx] = sum(gamma_[:severity_idx, subgroup_idx]);
        }
    }
}
model {
    // Specify priors
    for (exam_idx in 1:n_exams) {
        mu[exam_idx, 1] ~ normal(mu_mean, mu_sd);
        mu[exam_idx, 2] ~ normal(mu_mean, mu_sd);
    }
    for (subgroup_idx in 1:n_subgroups) {
        for (severity_idx in 1:n_severities-1) {
            gamma_[severity_idx, subgroup_idx] ~ normal(gamma_mean, gamma_sd);
        }
        for (reader_idx in 1:n_readers) {
            nu[reader_idx, subgroup_idx] ~ normal(nu_mean, nu_sd);
        }
        b[subgroup_idx] ~ normal(b_mean, b_sd);
    }

    // Specify model
    for (reader_idx in 1:n_readers) {
        for (severity_idx in 1:n_severities) {
            for (exam_idx in 1:n_exams) {
                if (mask[reader_idx, severity_idx, exam_idx] == 1) {
                    int subgroup_l = subgroups[exam_idx, 1];
                    int subgroup_r = subgroups[exam_idx, 2];
                    real theta_l = inv_logit(mu[exam_idx, 1] + gamma[severity_idx, subgroup_l] + nu[reader_idx, subgroup_l] + b[subgroup_l]);
                    real theta_r = inv_logit(mu[exam_idx, 2] + gamma[severity_idx, subgroup_r] + nu[reader_idx, subgroup_r] + b[subgroup_r]);
                    real pred_l = pred[reader_idx, severity_idx, exam_idx, 1];
                    real pred_r = pred[reader_idx, severity_idx, exam_idx, 2];
                    target += lmultiply(pred_l, theta_l) + lmultiply(1 - pred_l, 1 - theta_l);
                    target += lmultiply(pred_r, theta_r) + lmultiply(1 - pred_r, 1 - theta_r);
                }
            }
        }
    }
}
'''

@gin.configurable(module='probabilistic_inference')
def main(save_fpath,
         observed_pred_fpath,
         exam_info_fpath,
         priors,
         seed,
         n_posterior_samples):
    # Prepare data
    exam_info = pd.read_csv(exam_info_fpath)
    subgroups = get_subgroups(exam_info)
    n_subgroups = subgroups.max() + 1
    pred, mask = load_file(observed_pred_fpath)
    n_readers, n_severities, n_exams = mask.shape
    data = {
        'n_readers': n_readers,
        'n_severities': n_severities,
        'n_subgroups': n_subgroups,
        'n_exams' : n_exams,
        'n_sides' : len(Side),
        'mu_mean': priors[0],
        'gamma_mean': priors[1],
        'nu_mean': priors[2],
        'b_mean': priors[3],
        'mu_sd': priors[4],
        'gamma_sd': priors[5],
        'nu_sd': priors[6],
        'b_sd': priors[7],
        'subgroups': subgroups + 1,
        'pred': pred,
        'mask': mask}

    # Prepare posterior samples dict
    posterior_samples = {}
    for exam_idx in range(n_exams):
        posterior_samples[f'mu[{exam_idx + 1},{Side.LEFT.value + 1}]'] = []
        posterior_samples[f'mu[{exam_idx + 1},{Side.RIGHT.value + 1}]'] = []
    for subgroup_idx in range(n_subgroups):
        for severity_idx in range(n_severities):
            if severity_idx < n_severities - 1:
                posterior_samples[f'gamma_[{severity_idx + 1},{subgroup_idx + 1}]'] = []
            posterior_samples[f'gamma[{severity_idx + 1},{subgroup_idx + 1}]'] = []
        for reader_idx in range(n_readers):
            posterior_samples[f'nu[{reader_idx + 1},{subgroup_idx + 1}]'] = []
        posterior_samples[f'b[{subgroup_idx + 1}]'] = []

    # Run inference and populate posterior samples dict
    sm = pystan.StanModel(model_code=PGM_RADIOLOGISTS if 'radiologists' in observed_pred_fpath else PGM_DNNS)
    fit = sm.vb(data=data, seed=seed, output_samples=n_posterior_samples)
    for k in posterior_samples.keys():
        idx = fit['sampler_param_names'].index(k)
        posterior_samples[k] = np.array(fit['sampler_params'][idx])

    # Prepare posterior samples numpy arrays
    mu = np.full((n_exams, len(Side), n_posterior_samples), np.nan)
    gamma_ = np.full((n_severities - 1, n_subgroups, n_posterior_samples), np.nan)
    gamma = np.full((n_severities, n_subgroups, n_posterior_samples), np.nan)
    nu = np.full((n_readers, n_subgroups, n_posterior_samples), np.nan)
    b = np.full((n_subgroups, n_posterior_samples), np.nan)

    # Populate posterior samples numpy arrays
    for exam_idx in range(n_exams):
        mu[exam_idx, Side.LEFT.value] = posterior_samples[f'mu[{exam_idx+1},{Side.LEFT.value+1}]']
        mu[exam_idx, Side.RIGHT.value] = posterior_samples[f'mu[{exam_idx+1},{Side.RIGHT.value+1}]']
    for subgroup_idx in range(n_subgroups):
        for severity_idx in range(n_severities):
            if severity_idx < n_severities - 1:
                gamma_[severity_idx, subgroup_idx] = posterior_samples[f'gamma_[{severity_idx+1},{subgroup_idx+1}]']
            gamma[severity_idx, subgroup_idx] = posterior_samples[f'gamma[{severity_idx+1},{subgroup_idx+1}]']
        for reader_idx in range(n_readers):
            nu[reader_idx, subgroup_idx] = posterior_samples[f'nu[{reader_idx+1},{subgroup_idx+1}]']
        b[subgroup_idx] = posterior_samples[f'b[{subgroup_idx+1}]']
    save_file((mu, gamma_, gamma, nu, b), save_fpath)

if __name__ == '__main__':
    gin.parse_config_file(sys.argv[1])
    main()