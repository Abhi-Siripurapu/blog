October 21, 2022

## Introduction: Convergence of Biology and Machine Learning

Recently, I've been exploring synthetic biology. This project explores how we can engineer plasmids—circular DNA molecules—to control bacterial behavior for practical applications, like producing ethanol. But we speed everything up by an order of magnitude using AI.


---

## Step 1: Plasmid Design—The Foundation of Bioconversion

### The Role of Plasmids in Bacterial Programming

For those unfamiliar with plasmids, they are autonomous, self-replicating DNA molecules found in bacterial cells. Think of them as instruction manuals that bacteria can use to perform functions outside their natural repertoire. The plasmid I started with—**pUC-pdc-adh**—was designed to force *E. coli* bacteria to produce ethanol, but it came at a cost. The metabolic burden was too high, leading to bacterial die-off, which reduced overall yield.

**Goal:** Create a plasmid that optimized ethanol production while minimizing bacterial death. This required manipulating several factors:

- **Promoter regions**: Control how strongly and frequently genes are expressed.
- **Gene sequences**: Determine what proteins the bacteria produce (in this case, enzymes involved in ethanol production).
- **Ribosome binding sites (RBS)**: Fine-tune the translation of mRNA into proteins.
  
Initially, I relied a lot on guesswork, especially with promoter strength and gene arrangement.

---

## Step 2: Leveraging AI for Plasmid Optimization

### Why Neural Nets?

The relationship between plasmid structure and bacterial behavior is a **high-dimensional, nonlinear system**. Slight tweaks in gene arrangement, promoter strength, or ribosome binding site (RBS) strength can have unpredictable downstream effects on bacterial metabolism. A traditional algorithmic approach—using heuristics to optimize gene expression—didn't really work. I needed a model that could understand and predict the complex, nonlinear interactions between these genetic components.

Enter the neural network. Specifically, I used a **feedforward neural network** with multiple hidden layers, trained on experimental data gathered from various plasmid designs. The reason for using a neural network was simple: the system was too complex for traditional optimization methods. With its ability to generalize from data and make nonlinear predictions, the neural network could “learn” the relationships between plasmid design and bacterial performance.

### Paramaters


1. **Plasmid design parameters**:
   - Promoter strength (measured in relative units, RPU)
   - RBS efficiency
   - Gene arrangement (relative positions of the ethanol-production genes)
  
2. **Bacterial performance metrics**:
   - **Ethanol yield** (g/L)
   - **Bacterial survival rate** (percentage of living cells after 24, 48, and 72 hours)
   - **Metabolic stress** (estimated from oxygen uptake rate and byproducts like lactate)

The initial dataset came from lab experiments using a variety of plasmid designs. I ran these designs through a small bioreactor, measured ethanol output, and recorded bacterial survival. But lab data is noisy—random environmental variables, nutrient limitations, and even slight pipetting errors can introduce inconsistencies. To combat this, I employed **data augmentation techniques** that simulated noisy lab conditions, giving the neural network a more robust understanding of the system.

### Neural Network Architecture

I opted for a **3-layer feedforward neural network** with **ReLU activation functions** for nonlinearity. Each layer had the following configuration:

- **Input Layer (n=8)**: Encodes the 8 features of plasmid design (promoter strength, gene order, RBS efficiency, etc.).
- **First Hidden Layer (n=128)**: Captures the initial nonlinearity in gene-expression interactions. The number of neurons was determined through hyperparameter tuning (more on this later).
- **Second Hidden Layer (n=64)**: Refines the learned representations, focusing on the relationships between metabolic stress and ethanol yield.
- **Output Layer (n=2)**: Predicts ethanol yield and bacterial survival rate.

### Training Process and Loss Function

I used a **multi-objective loss function** that took into account both **ethanol production** and **bacterial viability**. Instead of treating these objectives separately, I combined them into a single weighted loss function:

$$

L = w_1 \cdot \text{MSE}(Y_{ethanol}, \hat{Y}_{ethanol}) + w_2 \cdot \text{MSE}(S_{bacteria}, \hat{S}_{bacteria})

$$
Where:
- $Y_{ethanol}$ is the actual ethanol yield, and $\hat{Y}_{ethanol}$ is the predicted yield.
- $S_{bacteria}$ is the actual survival rate, and $\hat{S}_{bacteria}$ is the predicted rate.
- $w_1$ and $w_2$ are weights that balance the importance of ethanol production versus bacterial survival. After some experimentation, I found that **w1 = 0.6** and **w2 = 0.4** yielded the best results, emphasizing ethanol yield without letting bacterial death spiral out of control.

This is actually really, really simple to build. That's intentional. More complex networks start to over-cook the entire optimization process. Overfitting happens really quickly here, so I wanted to keep the nn less powerful in general.

---

## Step 3: In Silico Testing of Plasmid Designs

Once the model was trained, I used it to predict the performance of thousands of virtual plasmid designs. This process was like “biological A/B testing.” I fed the neural network different configurations of promoter strengths, gene orders, and ribosome binding sites, and it spat out predictions for ethanol yield and bacterial survival.

### Gene Positioning

One of the most surprising insights was the impact of **gene positioning** on overall plasmid performance. The neural network uncovered a counterintuitive relationship: shifting one of the ethanol-producing genes downstream of the promoter, rather than placing it upstream as commonly done, led to a **20% increase in bacterial survival**. This was likely due to reduced transcriptional interference between adjacent genes—a phenomenon that’s hard to predict through intuition alone.

### Adaptive Promoter Strengths

Another significant discovery was the implementation of **adaptive promoters**—promoters that dynamically adjust their strength based on the bacterial metabolic state. The AI model suggested that a promoter which down-regulated gene expression under stress conditions resulted in healthier bacterial populations without sacrificing too much ethanol production. By reducing the metabolic load during critical periods (e.g., when the bacteria were low on glucose), the adaptive promoter system extended the lifespan of the bacterial culture, allowing for more efficient ethanol production over time.

---

## Step 4: Wet Lab Validation 

### Taking AI Predictions to the Bench

With the AI's top-performing plasmid designs, I synthesized these plasmids and introduced them into *E. coli* via electroporation. Each plasmid was tested under controlled conditions in a **bioreactor** to measure ethanol output and bacterial survival.

The AI-predicted plasmids performed exactly as expected. One particular design—featuring the adaptive promoter and shifted gene positions—yielded **25% more ethanol** than the baseline pUC-pdc-adh plasmid and extended bacterial survival by **30%**.

### Live Cultures

To make the system even more efficient, I embedded a **feedback loop** using a fluorescent reporter system. The plasmid was engineered with a secondary gene that encoded a fluorescent protein, which glowed when bacterial cells experienced metabolic stress. This allowed me to monitor bacterial health in real-time, adjusting the glucose feed rate and oxygenation in response to stress indicators, which kept the bacterial population healthier for longer periods.

---

## Conclusion: AI +Syn Bio:

I was able to create a bioconvertor that was significantly more efficient than the current state of the art (though still not economically viable).
The intersection of these two fields is still in its infancy. I think we need much better tooling/data collection methods to make building the models actually viable on a large scale.


---

