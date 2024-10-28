# CDRGen: Optimizing Antibody Binding with LLMs and CNNs
December 29, 2023

antibody optimization isn’t just a slow grind; it's a deeply iterative, *expensive* process, especially when we're zooming in on the CDR3 loop—the main site for antibody binding. with CDRGen, i’m sidestepping the usual wet lab slog using a neural-driven pipeline that leverages a **large language model (LLM)** to generate high-affinity antibody sequences and a **convolutional neural network (CNN)** to predict binding affinity. yeah, think of it as a kind of generative adversarial process for proteins.

let's dig into the **why** of each part of this architecture.

---

## background: why CDR3?

antibodies are like highly specific key-and-lock mechanisms for invaders (antigens). the main "key" part in antibodies is the **CDR3 loop** in the heavy chain. in most antibodies, **CDR1** and **CDR2** regions are encoded from germline genes, so they’re relatively predictable. but **CDR3?** it’s generated via genetic recombination, meaning it’s hyper-variable and carries massive potential for specific binding configurations. this is the site we want to refine.

### why not random mutagenesis?

traditional approaches use **mutagenesis** to create countless CDR3 variants, and then each variant is tested in the lab for binding affinity. problem is, this approach treats the sequence space as a random walk, sampling a mind-bendingly large number of configurations with low success rates. we wanted something *intelligent* that could explore this space in a guided way—hence, CDRGen.

---

## step 1: protein language model (PLM) for CDR3 generation

to generate optimized CDR3 sequences, i fine-tuned `AntiBERTA2`, a **protein language model (PLM)**. unlike typical transformers that train on text, AntiBERTA2 is trained on **protein sequences** where each amino acid is a token.

### why a transformer model?

transformers are built to predict the next token in a sequence based on context, which works beautifully for proteins where each amino acid’s position matters. but vanilla transformers don’t capture *structural information*—and for proteins, structure is key. AntiBERTA2 incorporates **contrastive language-image pretraining (CLIP)**, which means it updates token embeddings with extra structural information during training. this makes AntiBERTA2 ideal for protein design, as it gets a sense of which amino acids work together spatially as well as sequentially.

### why masked language modeling (MLM)?

to generate new CDR3 sequences, we used **masked language modeling (MLM)**. specifically, we masked the CDR3 region in our training data, forcing the model to "fill in the blanks" with amino acids that fit both the antibody and antigen context.

#### loss function:
$$
L_{\text{MLM}} = -\sum_{i=1}^{N} \log p(x_i | x_{\setminus i}; \theta)
$$
where:
- $x_i$ = the masked token to predict (amino acid),
- $x_{\setminus i}$ = context amino acids,
- $\theta$ = model parameters.

this training setup makes the model act like a guided “sequence completer” for any antibody input, filling in optimized CDR3s without randomization. 

---

## step 2: affinity prediction with CNN and normal mode analysis

to filter out weak binding candidates, i added a **convolutional neural network (CNN)** that predicts each generated sequence’s **binding affinity**. rather than running a wet lab affinity test, the CNN uses **normal mode analysis (NMA)** correlation maps, which capture the dynamical behavior of protein structures under small perturbations.

### why normal mode analysis (NMA)?

proteins aren’t static; they’re dynamic, constantly flexing and shifting. NMA breaks down this motion into "normal modes" that describe predictable fluctuations in structure. by generating **correlation maps** based on NMA, we get a structural “fingerprint” of how flexible or stable the generated CDR3 sequence is in binding context.

#### potential energy expansion:
$$
U(r) = U(r^*) + \frac{1}{2}(r - r^*)^T \nabla^2 U(r^*) (r - r^*) + \dots
$$
where $r^*$ is equilibrium position and $\nabla^2 U(r^*)$ is the **Hessian matrix**. we’re essentially describing the “elastic potential energy” between particles in the structure.

### cnn: choosing the architecture

the CNN reads these NMA-derived correlation maps $ X \in \mathbb{R}^{N \times N} $, where:
$$
X_{ij} \propto \sum_{l=0}^{L-1} \frac{a_{il} \cdot a_{jl}}{\omega(l)^2}
$$
- $a_{il}$ and $a_{jl}$ = amplitudes of fluctuations for residues $i$ and $j$,
- $\omega(l)$ = frequencies of normal modes.

cnn layer setup:
- **convolutional layers** with `kxk` filters capture spatial relationships in the correlation map, focusing on high-affinity clusters.
- **pooling layers** distill this information, highlighting the most significant affinity predictors.
- **output layer** predicts the log dissociation constant $K_D$, giving a quantitative measure of binding strength:
  
$$
\log_{10}(\hat{K}_D) = w^T z = \sum_{i=0}^{M-1} w_i z_i
$$

### why cnn for this task?

CNNs are naturally good at pattern recognition in spatial data. using NMA maps as input lets the CNN understand spatial relationships in protein structure, which directly influence binding affinity. it’s more efficient than sequence-based metrics and gets us from sequence to high-confidence predictions in silico.

---

## results: CDRGen performance and validation

we validated CDRGen’s performance on a test dataset, achieving a **92% improvement in binding affinity** across our sequences. here’s a performance comparison for SARS-COV-2 binding:

| Method       | Affinity Improvement | Aggregation Propensity |
|--------------|----------------------|-------------------------|
| EATLM        | 18%                 | -3%                   |
| OpenProtein  | 32%                 | -7%                   |
| ML-Guided    | 63%                 | -11%                  |
| **CDRGen**   | **108%**            | **-13%**              |

**aggregation propensity** measures the tendency of an antibody to clump, which is undesirable in therapeutic contexts. CDRGen’s design naturally selects against aggregating sequences, enhancing both **binding affinity and solubility**.

---

## future direction: stability, scalability, and production

CDRGen isn’t just a fast antibody optimizer; it’s a **platform** for on-demand therapeutic design. the next steps are:
- **predicting stability**: adding structural stability as a feature ensures antibodies remain functional under physiological conditions.
- **scaling production**: making sure generated antibodies are easy to produce in large quantities.

### multi-objective optimization:
in future iterations, we’ll add stability and scalability objectives using **multi-objective optimization**:
$$
L = \alpha L_{\text{affinity}} + \beta L_{\text{stability}} + \gamma L_{\text{scalability}}
$$
where alpha, beta, and gamma are weights to balance these priorities.

---

**tl;dr**: CDRGen uses a transformer + CNN setup to optimize antibodies faster, with better precision and lower aggregation, pushing us closer to rapid-response therapeutics.
