from code.utils import *

def plot_class_separability(ax, pred, exam_info, query, name):
    exam_idxs_l, exam_idxs_r = get_exam_idxs(exam_info, query)
    y = get_y(exam_info, exam_idxs_l, exam_idxs_r)
    ks = []
    n_readers, n_seeds, n_severities, n_exams, n_sides = pred.shape
    for reader_idx in range(n_readers):
        for seed_idx in range(n_seeds):
            ks_seed = np.full(len(SEVERITIES), np.nan)
            for severity_idx in range(n_severities):
                pred_l = pred[reader_idx, seed_idx, severity_idx, exam_idxs_l, 0]
                pred_r = pred[reader_idx, seed_idx, severity_idx, exam_idxs_r, 1]
                pred_lr = np.concatenate((pred_l, pred_r))
                pos_pred, neg_pred = pred_lr[y == 1], pred_lr[y == 0]
                ks_seed[severity_idx] = ks_2samp(pos_pred, neg_pred)[0]
            ks.append(ks_seed)
    ks = np.array(ks)
    mean = ks.mean(axis=0)
    sd = ks.std(axis=0)
    if name == 'ROI interior':
        x_offset = -0.2
    elif name == 'ROI exterior':
        x_offset = 0
    else:
        x_offset = 0.2
    ax.errorbar(np.arange(len(SEVERITIES)) + x_offset, mean, yerr=sd, marker='s', ls='none', label=name)
    p_values = []
    for severity_idx in range(1, len(SEVERITIES)):
        unperturbed = ks[:, 0]
        perturbed = ks[:, severity_idx]
        p_values.append(ks_2samp(unperturbed[~np.isnan(unperturbed)], perturbed[~np.isnan(perturbed)],
            alternative='less')[1])
    return p_values

@gin.configurable(module='annotation_study_analysis')
def main(save_fpath,
         observed_pred_dpath,
         exam_info_fpath,
         annotation_study_idxs_fpath,
         dnn_architecture,
         is_legend):
    exam_info = pd.read_csv(exam_info_fpath)
    annotation_study_idxs = load_file(annotation_study_idxs_fpath)
    exam_info = exam_info.iloc[annotation_study_idxs]
    pred_entire, _ = load_file(os.path.join(observed_pred_dpath, 'perturbation_study', 'dnns', dnn_architecture, 'unperturbed.pkl'))
    pred_entire = pred_entire[None, :, :, annotation_study_idxs, :]
    pred_interior = load_file(os.path.join(observed_pred_dpath, 'annotation_study', dnn_architecture, 'roi_interior.pkl'))
    pred_exterior = load_file(os.path.join(observed_pred_dpath, 'annotation_study', dnn_architecture, 'roi_exterior.pkl'))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    p_values_0 = []
    p_values_0.append(plot_class_separability(axes[0], pred_interior, exam_info, 'microcalcifications subtask', 'ROI interior'))
    p_values_0.append(plot_class_separability(axes[0], pred_exterior, exam_info, 'microcalcifications subtask', 'ROI exterior'))
    p_values_0.append(plot_class_separability(axes[0], pred_entire, exam_info, 'microcalcifications subtask', 'Entire image'))
    p_values_0 = np.array(p_values_0)

    p_values_1 = []
    p_values_1.append(plot_class_separability(axes[1], pred_interior, exam_info, 'soft_tissue_lesions subtask', 'ROI interior'))
    p_values_1.append(plot_class_separability(axes[1], pred_exterior, exam_info, 'soft_tissue_lesions subtask', 'ROI exterior'))
    p_values_1.append(plot_class_separability(axes[1], pred_entire, exam_info, 'soft_tissue_lesions subtask', 'Entire image'))
    p_values_1 = np.array(p_values_1)

    for ax in axes:
        ax.set_ylabel('Class separability (KS statistic)')
        ax.set_xlabel('Low-pass filter severity')
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels(SEVERITIES)
        ax.grid(True)

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
        ax.annotate('ROI int.', xy=(xfrac[0] - 0.029, 0.99), xycoords='figure fraction', horizontalalignment='left',
            verticalalignment='top', color=colors[0])
        ax.annotate('ROI ext.', xy=(xfrac[0] - 0.029, 0.94), xycoords='figure fraction', horizontalalignment='left',
            verticalalignment='top', color=colors[1])
        ax.annotate('Entire', xy=(xfrac[0] - 0.029, 0.89), xycoords='figure fraction', horizontalalignment='left',
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