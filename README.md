# Outlines
1. Implementation of ChiMerge (https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
  > Works `sklearn` way
2. Supervised discretization using `target`, `Chi2 statistics & test`
3. Can be configured to multiprocess (`n_jobs`)

```python
from discretization.chi_merge import *
chi_merge = ChiMerge(con_features=X.columns, significance_level=0.1, n_jobs=-3)
chi_merge.fit_transform(X)
```

# Concept
It follows below rules.
- If continuous feature is discretized,
    1. Within interval, class frequency is stable.
    2. Two adjacent intervals should have no similar class frequencies.
- This is tested by Chi2 test

# Random thoughts
- What if considering k-adjacent, not 2-adjacent ?
    > Should be normalized and reflected to formula in paper

# References
- ChiMerge: Discretization of Numeric Attributes (https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf (https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf))
- Discretization: An Enabling Technique (https://cs.nju.edu.cn/zhouzh/zhouzh.files/course/dm/reading/reading03/liu_dmkd02.pdf)

TODO  
- dataset to s3
