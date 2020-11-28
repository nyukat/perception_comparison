from code.utils import *

def plot_predictive_confidence(pgm_dpath, dnn_architecture, subgroup, ax, name):
    if name == 'Radiologists':
        pgm_fpath = os.path.join(pgm_dpath, 'radiologists.pkl')
        x_offset = -0.2
    elif name == 'DNNs':
        pgm_fpath = os.path.join(pgm_dpath, 'dnns', dnn_architecture, 'unperturbed.pkl')
        x_offset = 0
    else:
        pgm_fpath = os.path.join(pgm_dpath, 'dnns', dnn_architecture, 'filtered.pkl')
        x_offset = 0.2
    mu, gamma_, gamma, nu, b = load_file(pgm_fpath)
    category_idx = SUBGROUPS.index(subgroup)
    gamma = gamma[:, category_idx, :].squeeze()
    mean = gamma.mean(1)
    sd = gamma.std(1)
    p_values = []
    for severity_idx in range(1, len(SEVERITIES)):
        p_values.append(1 - scipy.stats.norm.cdf(0, mean[severity_idx], sd[severity_idx]))
    ax.errorbar(np.arange(len(SEVERITIES)) + x_offset, mean, yerr=sd, marker='s', ls='none', label=name)
    return p_values

def calc_class_separability(exam_info_fpath, subgroup, pred):
    exam_info = pd.read_csv(exam_info_fpath)
    exam_idxs_l, exam_idxs_r = get_exam_idxs(exam_info, subgroup + ' subtask')
    pred_l, pred_r = split_arr(pred)
    pred = np.concatenate((pred_l[exam_idxs_l], pred_r[exam_idxs_r]))
    y = get_y(exam_info, exam_idxs_l, exam_idxs_r)
    pos_pred, neg_pred = pred[y == 1], pred[y == 0]
    try:
        return ks_2samp(pos_pred, neg_pred)[0]
    except:
        return np.nan

def plot_class_separability(pgm_dpath, posterior_pred_dpath, exam_info_fpath, dnn_architecture, subgroup, ax, name):
    if name == 'Radiologists':
        pgm_fpath = os.path.join(pgm_dpath, 'radiologists.pkl')
        posterior_pred_fpath = os.path.join(posterior_pred_dpath, 'radiologists.pkl')
        x_offset = -0.2
    elif name == 'DNNs':
        pgm_fpath = os.path.join(pgm_dpath, 'dnns', dnn_architecture, 'unperturbed.pkl')
        posterior_pred_fpath = os.path.join(posterior_pred_dpath, 'dnns', dnn_architecture, 'unperturbed.pkl')
        x_offset = 0
    else:
        pgm_fpath = os.path.join(pgm_dpath, 'dnns', dnn_architecture, 'filtered.pkl')
        posterior_pred_fpath = os.path.join(posterior_pred_dpath, 'dnns', dnn_architecture, 'filtered.pkl')
        x_offset = 0.2
    if os.path.exists(posterior_pred_fpath):
        pred = load_file(posterior_pred_fpath)
    else:
        logging.info(f'Computing posterior predictions for {name}, this may take several minutes')
        pred = calc_posterior_pred(exam_info_fpath, pgm_fpath)
        save_file(pred, posterior_pred_fpath)
    n_readers, n_severities = pred.shape[:2]
    ks = np.full((n_readers, n_severities), np.nan)
    for reader_idx in range(n_readers):
        for severity_idx in range(n_severities):
            ks[reader_idx, severity_idx] = calc_class_separability(exam_info_fpath, subgroup, pred[reader_idx, severity_idx])
    ks = np.reshape(ks, (-1, n_severities))
    p_values = []
    for severity_idx in range(1, n_severities):
        clean = ks[:, 0]
        perturbed = ks[:, severity_idx]
        p_values.append(ks_2samp(clean[~np.isnan(clean)], perturbed[~np.isnan(perturbed)], alternative='less')[1])
    mean = np.nanmean(ks, axis=0)
    sd = np.nanstd(ks, axis=0)
    ax.errorbar(np.arange(n_severities) + x_offset, mean, yerr=sd, marker='s', ls='none', label=name)
    return p_values

@gin.configurable(module='perturbation_study_analysis')
def main(save_fpath,
         posterior_samples_dpath,
         posterior_pred_dpath,
         exam_info_fpath,
         dnn_architecture,
         subgroup,
         is_legend):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    p_values_0 = []
    p_values_0.append(plot_predictive_confidence(posterior_samples_dpath, dnn_architecture, subgroup, axes[0], 'Radiologists'))
    p_values_0.append(plot_predictive_confidence(posterior_samples_dpath, dnn_architecture, subgroup, axes[0], 'DNNs'))
    p_values_0.append(plot_predictive_confidence(posterior_samples_dpath, dnn_architecture, subgroup, axes[0], 'DNNs trained w/ filtered data'))
    p_values_0 = np.array(p_values_0)

    p_values_1 = []
    p_values_1.append(plot_class_separability(posterior_samples_dpath, posterior_pred_dpath, exam_info_fpath, dnn_architecture,
        subgroup, axes[1], 'Radiologists'))
    p_values_1.append(plot_class_separability(posterior_samples_dpath, posterior_pred_dpath, exam_info_fpath, dnn_architecture,
        subgroup, axes[1], 'DNNs'))
    p_values_1.append(plot_class_separability(posterior_samples_dpath, posterior_pred_dpath, exam_info_fpath, dnn_architecture,
        subgroup, axes[1], 'DNNs trained w/ filtered data'))
    p_values_1 = np.array(p_values_1)

    for ax in axes:
        ax.set_xlabel('Low-pass filter severity')
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels(SEVERITIES)
        ax.grid(True)

    axes[0].set_ylabel(r'$E[\gamma_{s, c} | \mathcal{D}]$')
    axes[1].set_ylabel('KS statistic')

    if is_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])

    tight_layout(fig)
    fig.subplots_adjust(top=0.84, bottom=0.25)

    for ax, p_values in zip(axes, [p_values_0, p_values_1]):
        line = ax.lines[1]
        x, y = line.get_data()
        xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
        xpix, ypix = xy_pixels.T
        width, height = fig.canvas.get_width_height()
        xfrac = xpix / width
        colors = [line.get_color() for line in ax.lines]
        ax.annotate('Radiol.', xy=(xfrac[0] - 0.029, 0.99), xycoords='figure fraction', horizontalalignment='left',
            verticalalignment='top', color=colors[0])
        ax.annotate('DNNs', xy=(xfrac[0] - 0.029, 0.94), xycoords='figure fraction', horizontalalignment='left',
            verticalalignment='top', color=colors[1])
        ax.annotate('DNNs (f)', xy=(xfrac[0] - 0.029, 0.89), xycoords='figure fraction', horizontalalignment='left',
            verticalalignment='top', color=colors[2])
        for severity_idx in range(1, len(SEVERITIES)):
            ax.annotate(f'{p_values[0, severity_idx - 1]:.3f}', xy=(xfrac[severity_idx], 0.99), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top', color=colors[0])
            ax.annotate(f'{p_values[1, severity_idx - 1]:.3f}', xy=(xfrac[severity_idx], 0.94), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top', color=colors[1])
            ax.annotate(f'{p_values[2, severity_idx - 1]:.3f}', xy=(xfrac[severity_idx], 0.89), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='top', color=colors[2])

    os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
    plt.savefig(save_fpath)

if __name__ == '__main__':
    gin.parse_config_file(sys.argv[1])
    main()