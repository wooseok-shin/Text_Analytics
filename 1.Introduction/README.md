# 1. Introduction to Text Analytics

## Part1

### 1) Text mining Overview

- Text Mining
	- 1) Unstructured Text Data로부터
	- 2) 다양한 방법론들을 사용해
	- 3) 유의미한 정보나 지식을 추출


- 예시 : 연설문 키워드 분석(워드 클라우드), 각국의 통화정책(Text문서) Cluster, 특허 분석(기존 특허와 얼마나 가까운지)
토픽 모델링(LDA, Topic name을 정해주는 것은 아직까지 사람이 한다), 스팸메일 필터링
감성분석(그 중에서도 어느 부분에 의해 긍정, 부정으로 판단했는지까지는 Attention 이전엔 활성화되지 않았었음)

- 한국어(단어 110만개, 너무 Sparse한 Vector가 되기때문에 비효율적)

- Text Structures
	- Weakly Structured (research papers)
	- Semi (E-mail, HTML/XML)

- 언어 자체의 모호성(ambiguous)이 Text Mining을 수행함에 있어 어려운 부분

- Text Mining Process
	- 1. 문제정의 및 텍스트 데이터 수집
	- 2. Preprocess 및 Transform the data (임의의 포맷(가변길이)을 특정한 고정길이의 벡터(ex.Bow)로 변환)
	- 3. Slect/Extract Features (고정길이의 벡터를 차원 축소)
	- 4. Algorithm Learning & Evaluation (2, 3에서 구조화시킨 데이터를 모델에 투입)



## Part2

- NLP 방법론들을 실험할 때 사용하는 Dataset
- ![방법론Dataset](https://user-images.githubusercontent.com/46666862/78104389-1a7bf080-742a-11ea-99ce-eac6b9f3f81b.PNG)


### 2) Text Preprocessing

- Level 0(Text)
	- 문서에서 이미지나 HTML같은 태그를 제외한 텍스트를 가져오는 것이 중요
	- meta-data(기사의 작성자, 날짜, 카테고리, 언어, 언론사 등의 정보)는 지워서는 안된다 --> 문서 분류나 시계열 분석을 할 때 유용하게 사용된다.

- Level 1(Sentence)
	- 문장을 올바르게 판별하는 것이 이후의 Task에 중요하다
		- POS-Tagger(형태소분석) maximize probabilities of tags within a sentence
		- Summarization systems rely on correct detection of sentence(문서에서 어떤 문장이 중요한지를 알아내는 것은 기본적으로 문장이 제대로 판별되었다고 가정을 깔고간다)
	- "." , "!" , "?"로 문장의 경계를 무조건적으로 판단할 수는 없다. ex) MR.Shin,  3.14, Y Corp.  etc..
	- 그리고 마침표가 없이 두 문장이 이어져 있는 경우에는 기계가 판단하지 못한다.

- Level 2 (Token)
	- 어떤 최소한의  의미를 지니는 단위 (단어, 숫자, 공백 등은 토큰으로 정의될 수 있다.)
	- 그러나 토큰 관점 역시, John's sick의 토큰이 하나인지 둘인지 정의하기 어렵다. ,  data-base,  C++, A/C와 같은 것들도..
	- 그럼에도 일관성있는 토큰화 작업은 매우 중요하다.
	- 관사같은 것들은 단어이지만 분석적인 관점에서 의미를 지니지 않으므로 제거(Stop-words)
	
	- Stemming
		- 기본 단어에서 과거, 미래 등으로 의미는 비슷하나 다른 형태를 지니는 단어들을 하나의 stem으로 Normalize form을 찾는 것 - 단어에서 같은 철자를 많이 포함하는 것이 base form이 된다.
		- Love, Loves, Loved, Loving --> Lov
		- Innovation, Innovations, Innovate, Innovates, Innovative --> Innovat
		- 기존에 존재하지 않는 단어일 수도 있지만 단어들을 원형으로 줄여나가기 때문에 Stem을 했을 때 결과물이 적은것이 장점
		- 차원축소적인 관점에서 효율적

	- Lemmatization
		- 해당하는 단어들이 가지고 있는 품사를 보존하면서 단어의 원형을 찾는 것
		- Love, Loves, Loved, Loving --> Love
		- Innovation, Innovations, Innovate, Innovates, Innovative --> Innovation, Innovation, Innovate, Innovate, Innovative
		- 단어의 품사를 보존하는 관점에서 효율적
		
	
### 3) Transformation

- Document Representation: 문서를 어떻게 연속형의 숫자 벡터로 표현할 것인지가 핵심
	- Bag-of-words: 문서안에서 단어의 빈도
	
	- Word Weighting(TF-IDF): 특정한 단어가 어떤 문서에서 얼마나 중요한지 (Term Frequency, Inverse Document Frequency)
		- tf(어떤 문서안에서 단어가 얼마나 등장했는지) X idf=log N/df( df = corpus안에서 몇번 등장했는지) 
		- 예를들어 the같은 관사는 모든 문서에서 1번씩은 등장했을 것, log N / N = log1이 되므로 TF-IDF값은 0이된다.
		- 즉, 어떤 단어가 중요하려면 A 문서안에서는 많이 등장하지만 다른 B, C, D...등의 문서에서는 적게 등장해야한다.

	- One-hot Vector Representation
		- Bag-of-words는 모두 원핫벡터로 표현되는 것
		- 원핫벡터로 표현이되면 단어간의 Similarity가 존재하지 않는다. (두 단어를 내적하면 0)
		- 따라서, 다른 표현방법을 찾자.
		
	- Word Vectors: Distribued Representation
		- W: words --> R^n   (n << 총 워드의 수)
		- W("cat") = (0.2, -0.4, 0.7, ...) 처럼 단어를 n차원의 실수로 표현해주는 것
		- King-Queen = Man-Woman  --> 성별이라는 것이 벡터로 표현됨 / 즉 단어 사이의 유사성이 보존됨
		- 최근엔, Elmo, GPT, BERT와 같은 Pre-trained Word Model 대신 Pre-trained Language Model을 사용
	
		
		
### 4) Dimensionality Reduction
- 차원 축소에는 크게 Feature Selection/Extraction이란 개념이 존재

- Feature Selection
	- Feature Subset Selection
		- 특정한 목적에 걸맞는 최적의 변수 집합을 **선택**하는 것, **주어진 변수를 가공하거나 변형시키지 않고 무엇이 중요할 것인지를 선택**, 즉 토큰같은 단위를 선택
		- 예를들어, 영화 리뷰 긍정/부정 분석에서 '핵노잼'이란 단어는 부정에 큰 영향을 미치는 단어가 되어 Information gain이 클 것이고, 영화 리뷰라는 단어는 긍/부정 어디든 많이 등장하므로 gain이 작다. 
		- 즉, Information gain, Cross-Entropy 와 같은 산출식들을 사용해서 어떤 단어나 토큰들이 우리가 원하는 Task(긍/부정 분석 등)에 유의미한지 아닌지를 판단하는 것이 Feature subset selection의 개념
		
- Feature Extraction
	- Feature Subset Extraction
		- 주어진 데이터로부터 새로운 변수를 **추출하거나 생성**하는 것
		- Extraction이 된 후의 차원은 기존의 차원보다 작아야 함
		- **기존 데이터의 정보는 최대한 보존하면서 차원을 줄이는 것이 핵심**
		- Latent Semantic Analysis(LSA)
		
	- Topic Modeling as a Feature Extractor
		- LDA, 토픽1~10에 각 문서가 얼만큼의 비중을 차지하고 있는지

	- Doc2Vec (Document차원에서 distributed representation을 찾는 것)
		- Word2Vec과 비슷하게 문서 자체를 단어와 함께 벡터공간에 맵핑


### 5) Learning & Evaluation

- Document Similarity
	- 유클리디안 보다는 코사인 유사도를 많이 사용한다.
	
- Document categorication(classification)
	- Spam filtering

- Sentiment Analysis
	- Sentiment tree bank (각 단어들별로 긍정 부정의 스코어가 매겨져있고, 트리 위로 올라오면서 최종적으로 긍정 부정인지 판단)

- Document Clustering & Visualization
	- 문서집합의 주요한 토픽을 찾는 것
	- 토픽들 사이의 관계식을 찾는 것
	- 어떤 일이 일어날지를 빠르게 파악

- Question Answering (SQuAD2.0 Dataset)









