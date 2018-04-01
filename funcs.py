from scipy import stats

def single_exp(x, scale):
    return stats.expon.pdf(x, loc = 0, scale = scale)