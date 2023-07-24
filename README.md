# TalkToEBM

> ### Did you ever want to talk to a model?

<img src="Spaceship%20Titanic_files/landing.png" alt="drawing" width="600"/>

TalkToEBM is an open-source package that provides a natural language interface to [Explainable Boosting Machines (EBMs)](https://github.com/interpretml/interpret). With this package, you can convert the graphs of Explainable Boosting Machines to text and generate prompts for LLMs. This package is currently under active development.

# Installation

Python 3.7+ | Linux, Mac, Windows
```sh
pip install t2ebm
```

# High-Level API: Pass the EBM to the LLM

```python
import t2ebm

gpt4 = guidance.llms.OpenAI("gpt-4")
```

We give have trained an ```ExplainableBoostingClassifier``` on the Kaggle [Spaceship Titanic Dataset](https://www.kaggle.com/competitions/spaceship-titanic/overview). The cool thing about this dataset is that its from 2022, so GPT-3.5 and GPT-4 have not seen it during training. 

```python
ebm = ExplainableBoostingClassifier(feature_names=feature_names)
ebm.fit(X_train, y_train)
```
Getting the LLM to describe a graph from the model is as simple as this

```python
graph_desc = t2ebm.llm_describe_ebm_graph(gpt4, ebm, 0) # feature 0, 'HomePlanet'
```

> **GPT-4:** *The feature "HomePlanet" significantly influences the model's outcome. Being
from "Europa" has the highest positive effect with a mean value of 0.87. The
"Mars" category also has a positive effect, albeit much smaller, with a mean
value of 0.098. On the contrary, being from "Earth" negatively impacts the
prediction, with a mean value of -0.439. The 95% confidence intervals suggest
that these effects are statistically significant and not due to random chance.
However, without additional context, it's difficult to determine why these
differences occur. Further investigation would be required for a deeper
understanding.*

We can give the LLM more context by providing a description of the dataset and the outcome. This allows the LLM to talk about the spacetime anomaly.

```python
dataset_description = """ The description of the dataset from Kaggle """

y_axis_description = """The y-axis depicts contributions in log-odds towards the probability
that the passenger was transported to another dimension."""
```


```python
graph_description = t2ebm.llm_describe_ebm_graph(gpt4, ebm, 0,                             
                                                 dataset_description=dataset_description,
                                                 y_axis_description=y_axis_description)     # again feature 0, 'HomePlanet'
```

> **GPT-4:** *The HomePlanet feature has a significant influence on the probability of a
passenger being transported to an alternate dimension during the collision with
the spacetime anomaly. Passengers from Europa have the highest positive effect,
indicating a higher likelihood of being transported. On the other hand,
passengers from Earth have a negative effect, suggesting a lower likelihood of
being transported. Interestingly, passengers from Mars have a smaller positive
effect compared to Europa, but still higher than Earth. This suggests that the
planet of origin plays a role in determining the probability of being
transported, with passengers from Europa having the highest likelihood. However,
there may be confounding factors or interactions with other features that
contribute to the observed pattern, and further analysis is needed to fully
understand the influence of the HomePlanet feature on the outcome.*

Instead of describing individual graphs, we can also ask the LLM to summarize the entire model.

```python
model_description = t2ebm.llm_describe_ebm(gpt4, ebm,
                                           dataset_description=dataset_description, 
                                           y_axis_description=y_axis_description) # by default this creates a 30 sentence summary
```

> **GPT-4:** *Based on the global feature importances and the summaries of the individual
features, we can provide the following summary:  The most important features in
the model are Spa, VRDeck, CryoSleep, and RoomService.   Passengers who spent
more at the spa had a lower probability of being affected by the anomaly. Higher
spending on the VRDeck also decreased the likelihood of being transported,
although there was a surprising counterintuitive behavior within a specific
spending range.   Choosing to be in cryosleep during the voyage increased the
probability of being affected by the anomaly, which was unexpected.   Higher
spending on room service corresponded to a higher probability of being affected,
but extremely high spending was associated with a lower probability.   Other
features had lower importances and may have less significant or less
straightforward effects on the outcome.  In summary, Spa, VRDeck, CryoSleep, and
RoomService were the most important features. The unexpected findings, such as
the counterintuitive influence of cryosleep and the complex relationships
observed in features like RoomService and VRDeck, warrant further investigation
and analysis.*

# Low-Level API: Extract Graphs from the EBM and perform custom prompts

```python
import t2ebm.graphs as graphs
```
We have a simple datastructure for graphs, and some useful operations on them

```python
graph = graphs.extract_graph(ebm, 9)  # feature 9, 'Spa'
graphs.plot_graph(graph)
```

<img src="Spaceship%20Titanic_files/Spaceship%20Titanic_21_0.png" alt="drawing" width="400"/>

The graphs learned by EBMs can contain many small details. We can simplify them to reduce the number of tokens. There is a parameter to control the degree of simplification.

```python
t2ebm.graphs.plot_graph(t2ebm.graphs.simplify_graph(graph, min_variation_per_cent=0.041))
```

<img src="Spaceship%20Titanic_files/Spaceship%20Titanic_24_0.png" alt="drawing" width="400"/>


The package also supports categorical and boolean features

```python
graphs.plot_graph(graphs.extract_graph(ebm, 0)) # home planet
```

<img src="Spaceship%20Titanic_files/Spaceship%20Titanic_22_0.png" alt="drawing" width="400"/>

We have to convert graphs to text before passing them to the LLM

```python
graph_as_text = graphs.graph_to_text(graph, max_tokens=1000)
print(graph_as_text)
```

>    This graph represents a continuous-valued feature. The keys are intervals that represent ranges where the function predicts the same value.
>    
>    Feature Name: Spa
>    Feature Type: continuous
>    Means: {"(0.0, 36.5)": 0.517, "(36.5, 117.5)": 0.212, ..., "(3978.0, 18572.0)": -5.917}
> 
>    Lower Bounds (95%-Confidence Interval): {"(0.0, 36.5)": 0.068, "(36.5, 117.5)": 0.049, ..., "(3978.0, 18572.0)": -6.956}
> > 
>    Upper Bounds (95%-Confidence Interval): {"(0.0, 36.5)": 0.966, "(36.5, 117.5)": 0.374, ..., "(3978.0, 18572.0)": -4.877}
    

Given the textual representation of the graph, you can start to write your own prompts. Usually, the first prompt will ask the LLM to describe the graph.


```python
prompt = t2ebm.prompts.describe_graph(graph_as_text,
                                      expert_description="{some kind of expert that will help me.}",
                                      y_axis_description="{The y-axis on the graph depicts how cool the graph is.}",
                                      special_task_description="{Pay special attention to run 'pip install t2ebm'}",
                                      dataset_description="{My dataset is so great, I wont't even tell you about it}",
                                      include_assistant_response=True)

# our prompts use guidance, and this is a nice way to print them
guidance(prompt, llm=gpt4)
```


> You are {some kind of expert that will help me.}
> 
> You interpret global explanations produced by a generalized additive model (GAM). GAMs produce explanations in the form of graphs that contain the effect of a specific input feature.
> 
> The user will first provide a general description of the dataset. Then you will be given graphs from the model, and the user will ask you questions about the graphs. 
> 
> Answer all questions to the best of your ability, combining both the data contained in the graph, the data set description you were given, and your knowledge about the real world.
> 
> Graphs will be presented as a JSON object with keys representing the x-axis and values representing the y-axis. For continuous features, the keys are intervals that represent ranges where the function predicts the same value. For categorical features, each key represents a possible value that the feature > > can take. {The y-axis on the graph depicts how cool the graph is.} 
> 
> The user will provide graphs in the following format:
>     - The name of the feature depicted in the graph
>    - The type of the feature (continuous, categorical, or boolean)
>     - Mean values
>     - Lower bounds of confidence interval
>     - Upper bounds of confidence interval
>
>{Pay special attention to run &#x27;pip install t2ebm&#x27;}
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{~/system}}</span>
>
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{#user~}}</span>
>{My dataset is so great, I wont&#x27;t even tell you about it}
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{~/user}}</span>
>
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{#assistant~}}</span>
>Thanks for this general description of the data set. Please continue and provide more information, for example about the graphs from the model.
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{~/assistant}}</span>
>
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{#user~}}</span>
>Consider the following graph from the model. This graph represents categorical feature. Each key represents a possible value that the feature can take.
>
>Feature Name: HomePlanet
>Feature Type: categorical
>Means: {&quot;Earth&quot;: -0.439, &quot;Europa&quot;: 0.87, &quot;Mars&quot;: 0.098}
>Lower Bounds (95%-Confidence Interval): {&quot;Earth&quot;: -0.475, &quot;Europa&quot;: 0.783, &quot;Mars&quot;: 0.034}
>Upper Bounds (95%-Confidence Interval): {&quot;Earth&quot;: -0.402, &quot;Europa&quot;: 0.957, &quot;Mars&quot;: 0.162}
>
>Please describe the general pattern of the graph.
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{~/user}}</span>
>
><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{#assistant~}}</span><span style='font-family: monospace; background-color: rgba(0, 0, 0, 0.05);'>{{gen &#x27;graph_description&#x27; temperature=0.7 max_tokens=2000}}</span><span style='font-family: monospace; background-color: >rgba(0, 0, 0, 0.05);'>{{~/assistant}}</span></pre></div>

# Citation

If you use this software in your research, please consider to cite our paper.

```bib
@inproceedings{lengerich2023llms,
  author    = {Benjamin J. Lengerich, Sebastian Bordt, Harsha Nori, Mark E. Nunnally, Yin Aphinyanaphongs, Manolis Kellis, and Rich Caruana},
  title     = {LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs},
  booktitle = {arxiv},
  year      = {2023}
 }
```
