"""第 22 章快速测试：统计区间、配对差、分桶、错例标签、近重复与窥视模拟；不需要 GPU 或模型权重。"""
import numpy as np

from miniocr.eval.analysis import (bootstrap_mean, error_class, formula_bucket, length_bucket,
                                   near_overlap, paired_bootstrap_diff, peeking_simulation)


def test_bootstrap_interval_contains_mean_and_is_seeded():
    a = bootstrap_mean([0, 1, 1, 0, 1, 1, 1, 0], samples=500, seed=3)
    b = bootstrap_mean([0, 1, 1, 0, 1, 1, 1, 0], samples=500, seed=3)
    assert a == b
    assert a['interval'][0] <= a['value'] <= a['interval'][1]
    assert bootstrap_mean([])['value'] is None


def test_paired_difference_of_identical_outputs_is_zero():
    d = paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1], samples=200)
    assert d['diff'] == 0 and d['interval'] == (0.0, 0.0)


def test_paired_difference_direction():
    d = paired_bootstrap_diff([0, 0, 0, 0], [1, 1, 1, 1], samples=200)
    assert d['diff'] == 1 and d['interval'] == (1.0, 1.0) and d['p_b_not_better'] == 0.0


def test_buckets():
    assert formula_bucket(r'\dfrac{1}{2}') == '分式'
    assert formula_bucket(r'\begin{bmatrix}1\end{bmatrix}') == '矩阵/数组'
    assert formula_bucket('x_1') == '上下标'
    assert formula_bucket('x+1') == '其他'
    assert length_bucket('a' * 29) != length_bucket('a' * 30)
    assert length_bucket('a' * 60) == '长(>=60字符)'


def test_error_labels_cover_format_and_unparsable():
    assert error_class('x^{2}', 'x^{2}', 'x^{2}', is_clean=True) == 'correct'
    assert error_class('x^{2}', 'x_{2}', 'The answer is x^{2}', is_clean=False) == 'format'
    assert error_class('x^{2}', 'x_{2}', '\\frac{1', is_clean=False) == 'unparsable'


def test_near_overlap_finds_exact_copy_and_not_unrelated():
    ref = ['\\frac{a}{b}+c', '\\sum_{i=1}^{n} x_i', '\\alpha\\beta\\gamma']
    q = ['\\frac{a}{b}+c', '\\int_0^1 f(x)dx']
    out = near_overlap(ref, q, threshold=.8)
    assert out['hits'] == 1 and out['checked'] == 2
    assert out['pairs'][0][0] == 0


def test_peeking_bias_grows_with_looks():
    sim = peeking_simulation(true_accuracy=.6, trials=1000, seed=1)
    rows = sim['rows']
    assert abs(rows[0]['val_select_mean'] - .6) < .03
    assert abs(rows[0]['optimism']) < .03 and rows[-1]['optimism'] > rows[0]['optimism'] + .02
    assert np.isfinite([r['test_best_of_k_mean'] for r in rows]).all()
