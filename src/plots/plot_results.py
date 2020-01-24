import pandas as pd
import numpy as np
import plotnine
import statsmodels.formula.api as smf

from tabulate import tabulate


def load_data(filter_methods=None):
    df = pd.read_csv('../../results/transfer.csv', index_col=0)
    df['percent_broken'] = 1 - df['percent_broken']
    df['percent_broken'] = (df['percent_broken'] * 100)
    df['percent_fail_runs'] = (df['percent_fail_runs'] * 100)
    df['log_score'] = df['rul_score'].apply(np.log)
    task = df.index.map(lambda x: x.split('_')[1])
    method = df.index.map(lambda x: x.split('_')[2])
    df = df.assign(task=task.values)
    df = df.assign(method=method.values)

    if filter_methods is not None:
        df = df[df['method'].isin(filter_methods)]

    baseline = pd.read_csv('../../results/baseline.csv', index_col=0)
    baseline = pd.DataFrame(baseline.stack())
    baseline.columns = ['value']
    baseline['dataset'] = baseline.index.labels[0]
    baseline['measure'] = baseline.index.labels[1]
    baseline['type'] = baseline['measure'] > 3
    baseline['measure'] = baseline['measure'] % 4
    baseline.loc['cmapss_one_baseline', 'dataset'] = 0
    baseline.loc['cmapss_two_baseline', 'dataset'] = 1
    baseline.loc['cmapss_three_baseline', 'dataset'] = 2
    baseline.loc['cmapss_four_baseline', 'dataset'] = 3

    baseline_cross = baseline[baseline['dataset'] != baseline['measure']]
    baseline_cross = baseline_cross[(baseline_cross['dataset'] != 0) | (baseline_cross['measure'] != 3)]
    baseline_cross = baseline_cross[(baseline_cross['dataset'] != 1) | (baseline_cross['measure'] != 2)]
    baseline_cross = baseline_cross[(baseline_cross['dataset'] != 2) | (baseline_cross['measure'] != 1)]
    baseline_cross = baseline_cross[(baseline_cross['dataset'] != 3) | (baseline_cross['measure'] != 0)]

    task_dict = {20: 'three2one', 1: 'one2two', 2: 'one2three', 13: 'two2four',
                 10: 'two2one', 31: 'four2two', 32: 'four2three', 23: 'three2four'}
    baseline_cross['task'] = 10 * baseline_cross['dataset'] + baseline_cross['measure']
    baseline_cross['task'] = baseline_cross['task'].apply(lambda x: task_dict[x])

    baseline_rul = baseline_cross[baseline_cross['type'] == 0]
    baseline_mse = baseline_cross[baseline_cross['type'] == 1]

    baseline_rul['log_value'] = baseline_rul['value'].apply(np.log)

    df['task'] = df['task'].map({'four2three': '4→3', 'four2two': '4→2', 'one2three': '1→3',
                                 'one2two':    '1→2', 'three2four': '3→4', 'three2one': '3→1',
                                 'two2four':   '2→4', 'two2one': '2→1'})
    baseline_mse['task'] = baseline_mse['task'].map({'four2three': '4→3', 'four2two': '4→2', 'one2three': '1→3',
                                                     'one2two':    '1→2', 'three2four': '3→4', 'three2one': '3→1',
                                                     'two2four':   '2→4', 'two2one': '2→1'})
    baseline_rul['task'] = baseline_rul['task'].map({'four2three': '4→3', 'four2two': '4→2', 'one2three': '1→3',
                                                     'one2two':    '1→2', 'three2four': '3→4', 'three2one': '3→1',
                                                     'two2four':   '2→4', 'two2one': '2→1'})

    return df, baseline_rul, baseline_mse


def mixed_linear_plots(df, x_axis, x_label):
    plotnine.options.figure_size = (8, 10)

    md = smf.mixedlm('log_score ~ percent_broken + percent_fail_runs', df, groups=df.index.values)
    mdf_rul = md.fit()

    print('#' * 18 + 'Log RUL' + '#' * 18)
    print(mdf_rul.summary())

    md = smf.mixedlm('mse ~ percent_broken + percent_fail_runs', df, groups=df.index.values)
    mdf_mse = md.fit()

    print('#' * 18 + 'RMSE' + '#' * 18)
    print(mdf_mse.summary())

    df['percent_broken'] = df['percent_broken'].round().astype(np.int)
    df['percent_fail_runs'] = df['percent_fail_runs'].round().astype(np.int)

    gg = (plotnine.ggplot(df, plotnine.aes(x=x_axis, y='log_score', color='method'))
          + plotnine.geom_jitter(width=2.5, show_legend=False)
          + plotnine.geom_abline(plotnine.aes(intercept=mdf_rul.params['Intercept'], slope=mdf_rul.params[x_axis]))
          + plotnine.stat_smooth(method='gls', show_legend=False)
          + plotnine.xlab(x_label)
          + plotnine.ylab('Logarithmic RUL-Score')
          + plotnine.scale_color_discrete(name='Method', labels=['DAAN', 'JAN'])
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_log_rul_by_method.pdf' % x_axis)

    gg = (plotnine.ggplot(df, plotnine.aes(x=x_axis, y='log_score', color='task'))
          + plotnine.geom_jitter(width=2.5, show_legend=False)
          + plotnine.geom_abline(plotnine.aes(intercept=mdf_rul.params['Intercept'], slope=mdf_rul.params[x_axis]))
          + plotnine.stat_smooth(method='gls', show_legend=False)
          + plotnine.xlab(x_label)
          + plotnine.ylab('Logarithmic RUL-Score')
          + plotnine.scale_color_discrete(name='Task', labels=['4→3', '4→2', '1→3', '1→2', '3→4', '3→1', '2→4', '2→1'])
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_log_rul_by_task.pdf' % x_axis)

    gg = (plotnine.ggplot(df, plotnine.aes(x=x_axis, y='mse', color='method'))
          + plotnine.geom_jitter(width=2.5)
          + plotnine.geom_abline(plotnine.aes(intercept=mdf_mse.params['Intercept'], slope=mdf_mse.params[x_axis]))
          + plotnine.stat_smooth(method='gls')
          + plotnine.ylab('RMSE')
          + plotnine.xlab(x_label)
          + plotnine.scale_color_discrete(name='Method', labels=['DAAN', 'JAN'])
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_mse_by_method.pdf' % x_axis)

    gg = (plotnine.ggplot(df, plotnine.aes(x=x_axis, y='mse', color='task'))
          + plotnine.geom_jitter(width=2.5)
          + plotnine.geom_abline(plotnine.aes(intercept=mdf_mse.params['Intercept'], slope=mdf_mse.params[x_axis]))
          + plotnine.stat_smooth(method='gls')
          + plotnine.ylab('RMSE')
          + plotnine.scale_color_discrete(name='Task', labels=['4→3', '4→2', '1→3', '1→2', '3→4', '3→1', '2→4', '2→1'])
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_mse_by_task.pdf' % x_axis)


def mixed_linear_factors_plot(df, x_axis, factor):
    plotnine.options.figure_size = (10, 10)
    factor_steps = df[factor].unique()
    reg_lines = pd.DataFrame({factor: factor_steps,
                              'intercept': np.zeros_like(factor_steps),
                              'slope': np.zeros_like(factor_steps)})
    for i, step in enumerate(factor_steps):
        factored_df = df[df[factor] == step]
        md = smf.mixedlm('mse ~ %s' % x_axis, factored_df, groups=factored_df.index.values)
        mdf = md.fit()
        reg_lines.iloc[i] = [step, mdf.params['Intercept'], mdf.params[x_axis]]

    df['percent_broken'] = df['percent_broken'].round().astype(np.int)
    df['percent_fail_runs'] = df['percent_fail_runs'].round().astype(np.int)
    reg_lines[factor] = reg_lines[factor].round().astype(np.int)
    gg = (plotnine.ggplot(df, plotnine.aes(x=x_axis, y='mse', color='method'))
          + plotnine.geom_jitter(width=2.5, show_legend=False)
          + plotnine.scale_color_manual(['#DB5F57'] * 4)
          + plotnine.facet_wrap(factor)
          + plotnine.geom_abline(plotnine.aes(intercept='intercept', slope='slope'), data=reg_lines)
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_vs_%s_rmse.pdf' % (x_axis, factor))


def method_plot(df, baseline_rul, baseline_mse, method):
    plotnine.options.figure_size = (15, 8)

    jan = df[df['method'] == method]

    jan['percent_broken'] = jan['percent_broken'].round().astype(np.int)
    jan['percent_fail_runs'] = jan['percent_fail_runs'].round().astype(np.int)

    plotnine.ylim = (2, 10)
    gg = (plotnine.ggplot(jan, plotnine.aes(x='percent_broken', y='log_score', color='method'))
          + plotnine.facet_wrap('task', 2, 4)
          + plotnine.stat_boxplot(plotnine.aes(y='log_value', x=60), data=baseline_rul, width=80, color='#14639e',
                                  show_legend=False)
          + plotnine.geom_jitter(width=2.5, show_legend=False)
          + plotnine.stat_smooth(method='gls', show_legend=False)
          + plotnine.xlab('Grade of Degradation in %')
          + plotnine.ylab('Logarithmic RUL-Score')
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_log_rul.pdf' % method)

    plotnine.ylim = (90, 10)
    gg = (plotnine.ggplot(jan, plotnine.aes(x='percent_broken', y='mse', color='method'))
          + plotnine.facet_wrap('task', 2, 4)
          + plotnine.stat_boxplot(plotnine.aes(y='value', x=60), data=baseline_mse, width=80, color='#14639e',
                                  show_legend=False)
          + plotnine.geom_jitter(width=2.5, show_legend=False)
          + plotnine.stat_smooth(method='gls', show_legend=False)
          + plotnine.xlab('Grade of Degradation in %')
          + plotnine.ylab('RMSE')
          + plotnine.theme_classic(base_size=20)
          )
    gg.save('%s_rmse.pdf' % method)


def method_tables(df: pd.DataFrame, baseline_rul: pd.DataFrame, baseline_mse, method):
    jan = df[df['method'] == method]
    jan.loc[:, 'percent_broken'] = jan['percent_broken'].round().astype(np.int)
    jan.loc[:, 'percent_fail_runs'] = jan['percent_fail_runs'].round().astype(str)

    jan = jan.drop(columns=['log_score', 'percent_fail_runs'])
    means = jan.groupby(['task', 'percent_broken']).mean()
    stds = jan.groupby(['task', 'percent_broken']).std()
    rul = means['rul_score'].combine(stds['rul_score'],
                                                    func=lambda mean, std: '$%.2f \pm %.2f$' % (mean, std))
    mse = means['mse'].combine(stds['mse'],
                                        func=lambda mean, std: '$%.2f \pm %.2f$' % (mean, std))
    rul = rul.unstack(level=-1).sort_index(level=1, axis=1)
    mse = mse.unstack(level=-1).sort_index(level=1, axis=1)

    baseline_rul = baseline_rul.drop(columns=['dataset', 'measure', 'type'])
    baseline_rul_means = baseline_rul.groupby('task').mean()
    baseline_rul_stds = baseline_rul.groupby('task').std()
    baseline_rul = baseline_rul_means['value'].combine(baseline_rul_stds['value'],
                                                       func=lambda mean, std: '$%.2f \pm %.2f$' % (mean, std))
    baseline_rul = pd.DataFrame(baseline_rul)
    baseline_rul.columns = ['source only']
    rul = pd.concat([baseline_rul, rul], axis=1)

    baseline_mse = baseline_mse.drop(columns=['dataset', 'measure', 'type'])
    baseline_mse_means = baseline_mse.groupby('task').mean()
    baseline_mse_stds = baseline_mse.groupby('task').std()
    baseline_mse = baseline_mse_means['value'].combine(baseline_mse_stds['value'],
                                                       func=lambda mean, std: '$%.2f \pm %.2f$' % (mean, std))
    baseline_mse = pd.DataFrame(baseline_mse)
    baseline_mse.columns = ['source only']
    mse = pd.concat([baseline_mse, mse], axis=1)

    print('RUL:')
    print(tabulate(rul, headers='keys', tablefmt='latex_raw'))
    print()
    print('RMSE:')
    print(tabulate(mse, headers='keys', tablefmt='latex_raw'))


if __name__ == '__main__':
    transfer, base_rul, base_rmse = load_data(filter_methods=['dadv', 'jan'])
    mixed_linear_plots(transfer, 'percent_broken', 'Grade of Degradation in %')
    mixed_linear_plots(transfer, 'percent_fail_runs', 'Number of Systems in %')
    mixed_linear_factors_plot(transfer, 'percent_fail_runs', 'percent_broken')
    mixed_linear_factors_plot(transfer, 'percent_broken', 'percent_fail_runs')
    for m in ['dadv', 'jan']:
        method_plot(transfer, base_rul, base_rmse, m)
    for m in ['dadv', 'jan']:
        print('')
        print('#' * 18 + m + '#' * 18)
        print('')
        method_tables(transfer, base_rul, base_rmse, m)
