# CosmicBLUP 中文用户指南

## 1. 定位

CosmicBLUP 专注于 BLUP 育种值估计，不包含 Bayes 或变分 Bayes 回归。主流程为关系矩阵构建、表型与固定效应装载、方差组分估计、混合模型方程求解和结果输出。

## 2. PBLUP

```bash
cosmicblup --blup --model pblup \
  --ped pedigree.txt --pheno phenotype.txt --pheno-name trait \
  --vce --vce-mode ai --threads 4 --out results/pblup
```

## 3. GBLUP

```bash
cosmicblup --blup --model gblup \
  --bfile genotype_prefix --pheno phenotype.txt --pheno-name trait \
  --vce --vce-mode ai --out results/gblup
```

`--pfile` 可用于 PGEN 输入。普通显式 GBLUP 尚不支持 BGEN；BGEN 当前用于 QC 和部分 STCG 路径。

## 4. ssGBLUP

```bash
cosmicblup --blup --model ssgblup \
  --ped pedigree.txt --bfile genotype_prefix \
  --pheno phenotype.txt --pheno-name trait \
  --vce --vce-mode ai --out results/ssgblup
```

谱系 ID 与基因型样本 ID 必须一致。程序会构造 A、G、A22 及组合 H 逆矩阵。

## 5. 固定方差求解

```bash
cosmicblup --blup --model gblup --bfile genotype_prefix \
  --pheno phenotype.txt --pheno-name trait \
  --skip-vce --var-a 0.4 --var-e 0.6 --out results/fixed
```

`--var-a` 与 `--var-e` 必须同时提供且为正数。

## 6. 重复力与随机回归

重复力模型使用 `--pe` 添加永久环境效应。随机回归使用 `--rrm`、`--time-col` 和 `--rrm-order`：

```bash
cosmicblup --rrm --ped pedigree.txt --pheno longitudinal.txt \
  --pheno-name trait --time-col age --rrm-order 1 --pe \
  --vce --vce-mode exact --out results/rrm
```

## 7. 多性状

```bash
cosmicblup --multi-trait --model gblup --bfile genotype_prefix \
  --pheno phenotype.txt --pheno-name trait1,trait2 \
  --vce --vce-mode ai --out results/multitrait
```

当前多性状流程面向稠密 GBLUP，并使用完整观测个体。

## 8. 关系矩阵

```bash
cosmicblup --make-matrix --relationship g --bfile genotype_prefix \
  --output-format bin --out matrices/g
```

文本矩阵使用 1-based 数字索引三元组，样本顺序写入独立 `.id` 文件。二进制格式为带元数据头的 `COSMIC_UPPER_PACKED`。

## 9. 输出

常规 BLUP 输出包括 `.vars`、`.beta`、`.rand`、`.obsrand`、`.anova` 和 `.log`。具体字段见 `OUTPUT_FORMATS.md`。
