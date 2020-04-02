# 2. Text Preprocessing

## Part1

### 1) Introduction to NLP

- NLP 예시
	- Lexical Analysis : A teacher come+s (관사+명사+동사+s 라는 분석결과물)
	- Syntax Analysis : (A teacher)_NP  (comes)VP   = A teacher는 명사구이면서 주어부, comes는 동사구이면서 술어부
	- Semantic Analysis : exist(x , teacher(x), comes(x)) = 의미까지 파악, teacher라는 객체가 존재하고 그 객체가 현재 오고(comes)있다
	- 위 세단계는 현재까지 할 수 있는 수준
	- Pragmatic Analysis : "A teacher comes"(떴다) = Be quite! , 함축된 뜻까지 파악하는 것 --> 현재 컴퓨터로는 불가능 (헤밍웨이 6words)
	
- 좁은 의미에서의 NLP도 어려운 이유
	- 프로그래밍 언어에 비해서 매우 많은 단어가 존재
	- 짱, 캡과 같은 단어는 이제는 쓰지않음. 즉, 시간에 따라서 단어가 생기고 사라지고 변화하기 때문에 어렵다.
	- 모호성(Ambiquity), He saw the man with the telescope

- Research Trends in NLP
	- 1. Rule-based 접근 (언어의 동적 특성때문에 힘듬, 하지만 여전히 많이 쓰이기도 함)
	- 2. Statistical 접근 (Hidden Markov Models, SVM 등, 룰베이스와 섞어서 많이 사용)
	- 3. Machine Learning(Deep) 접근, 연역적에서 귀납적 사고 로 메타가 바뀌게 됨

- 딥러닝으로 들어오면서 End-to-End Multi Task Learning 가 중요해짐

- Data Annotation (라벨링)
	- AWS(데이터 Make), DataMaker, 테스트웍스


### 2) Lexical Analysis(어휘분석)

- 목적 : 어떤 일정한 순서가 있는 캐릭터들의 조합을 토큰으로 변환하는 것. 즉, 의미를 보존할수있는 최소한의 수준에서 분석하는 것
	- 보통은 형태소를 토큰으로, 또는 단어를 토큰으로 사용함 (한글같은 경우에도 용언뒤 조사어미의 변화형은 형태소 관점에서는 의미가 있을 수 있지만 semantic관점에서는 의미가 없다고 판단하는데 그때는 단어 단위로)
	
- Process
	- 1. Tokenizing
	- 2. Part-of-Speech(POS) tagging : 각각의 문장이 어떤 형태소를 가지는지 태깅
	- 3. 추가적으로 필요하다면 개체명인지, 사람인지, 장소인지, 시간인지, 명사구인지 등을 판단
		

- Analysis
	- 1. Sentence Splitting
		- 문장을 구분하는 것이 NLP에서 중요하지만 일부 텍스트 마이닝에서는 중요하지 않을 수 있음(ex.토픽 모델링)
		- 문장을 구분하는 것 자체도 어렵긴 하다. (룰베이스 또는 분리된 문장들을 기준으로 학습을 시키는 방법)
	
	- 2. Tokenization
		- 문서를 의미를 지니는 기본 유닛인 토큰으로 변환
		- 단어, 숫자, space token으로 변환 : 토크나이저 모델마다 Number는 제거, 또는 제거하지 않는 것이 다름 --> 숫자가 중요한 의미를 가지는 Task의 경우는 제거하지 않는 Tokenizer를 사용해야 함
		- 토큰화 자체도 쉽지는 않다. (하이푼, 중국어 등)

	- 3. Morphological Analysis (형태소 분석)
		- 형태소의 Variants(변형성)이 존재
		- 현재까지는 이러한 변형성을 어떠한 Normal한 형태의 단어를 찾으면서 차원을 줄이는 방향으로 연구가 진행되어 왔음
		- 그 중 대표적인것이 Stemming과 Lemmatization
			- Stemming: Information Retrieval(정보추출)의 관점에서 주로 사용됨
				- Rule-based 알고리즘
				- 간단하고 빠른것이 장점
				- 그러나, Rule이 언어에 종속적이다.
				- 그리고 생성된 언어가 기존 언어에 존재하지 않는 언어일 수 있다. (computers -> comput)
				- 서로 다른 단어가 하나의 stem으로 바뀔 수 있음 (army, arm --> arm  / stocks, stockings --> stock)
				- 이는, 뒷단에 수행해야하는 semantic 분석관점에서 문제가 발생할 수 있음
				
			- Lemmatization: 
				- 실질적으로 존재하는 단어(품사까지 보존, root form 기반)
				- stemming보다 오류가 작다.
				- 그러나, 알고리즘이 더 복잡하고 느리다.
				
			- Information Retrieval 또는 Te xt Mining에서 어떤 것을 사용해야 하는가?
				- Semantic 분석(의미 분석)이 중요한 경우에는 Lemmatization을 사용
				- Semantic 분석(의미 분석)이 중요하지 않으면 Stemming을 사용
				
				
	- 4. Part-of-Speech(POS) Tagging  (품사 태깅)
		- 문장 X가 주어졌을 때, 그 문장의 단어(또는 Token)들에 대응하는 POS 태그를 찾아주는 것
		
		- ![2-1 사진](https://user-images.githubusercontent.com/46666862/78104372-19e35a00-742a-11ea-8c53-d84d74cc0e59.PNG)
		
		- 같은 단어나 토큰이라도 상황에 맞게 품사를 태깅해주는 것이 POS 태깅의 목적
		- Tagsets은 각각다르고(세종품사태그, 꼬꼬마 태그) 그에따라 결과물도 달라질 수 있음
		
		- POS Tagging을 위한 알고리즘 
			- Training Phase에서 직접 모든 품사들이 태깅되어 있지 않으므로 머신러닝 알고리즘이나 학습되어진 태깅 알고리즘을 사용한다. 이 때 일반적으로 좋은(예시로 96%) 성능을 가지는 것을 사용하지만 이는 모델을 학습할 때 사용되는 것과 같은 Domain에서만 그 성능유지를 보장할 수 있다. (의료 용어로 학습시키고 다른 구어체에 적용 시 태깅의 정확도가 낮아짐)
			- Decision Tree, HMMs, SVM 알고리즘 등을 사용하였었음. (딥러닝 이전)
			- 요즘에는 Transformation-based Taggers (e.g. , the brill tagger)
			
			- 1) Pointwise Prediction: classifier를 가지고 각각의 단어의 품사를 개별적으로 예측하는 것 (Maximum Entropy Model, SVM)
				- **예측에 주위 몇 가지 단어만을 가지고 함**
				- ex) Natural language [  ] (NLP) is a ~~ 이라는 문장이 있을 때 중간의 []에 어떤 단어가 들어가는지를 예측하는 것
				- 작동방식
					- 태그 예측을 하기위해 피쳐를 Encoding 해야 함: Suffix, prefix, 주변 단어의 정보를 토대로 변환
					- e.g. : 1 if suffix(w_j) = 'ing' & t_j = VBG, else 0
			
			- Probabilistic Models (**문장 전체를 입력으로 받음** )
				- 2) Generative sequence models (HMM): word1, word2, word3 순차적으로 품사를 예측
					- 작동방식
						- 전체 문장관점에서 봤을 때 확률을 최대로 하는 태그들의 조합을 찾는 것 
						- argmax P(Y|X)= argmax P(X|Y) x P(Y) --> P(X|Y)는 word와 POS사이(이러한 단어는 어떤 품사를 가질것이다.)의 관계를 확률로, P(Y)는 POS와 POS사이의 관계(관사다음에는 명사가 올 확률이 높다 등)를 확률로 하여 두가지를 고려해 판단하도록
					
				- 3) Discriminative sequence models (CRF): word1, word2, .. , wordn 모두를 한 번에 예측
					- 뉴럴넷 기반이 많이 출현했으나 여전히 POS Tagging 분야에서는 CRF알고리즘이 SOTA를 찍고 있다. 
					- 작동방식
						- 문장전체를 한번에 예측
			
			- 4) Neural network 기반 models (BERT etc)
				- Window-based vs Sentence-based
				- RNN (Many to Many): 단어 단위로 들어갈 수도 있고, 음절(Character)단위로도 들어갈 수 있음
			
			- 5) Hybrid Model: LSTM + ConvNet + CRF
			
			
	- 5. Named Entity Recognition (NER, 개체명 인식)
		- 어떤 문장들이 주어졌을 때 각각의 element가 미리 정의된 카테고리에 할당을 해주는 것
		- ex) "Get me a flight from New York City to San Francisco for next Thursday." 가 있을 때 "어디로 가느냐"에 대한 질문에는 SF로 간다, "언제가느냐"는 Thurday와 같이 어떤 물음에 대해 적절한 답변을 해주기 위한 방법 (챗봇과 같은 Task에서 중요)
		- 작동방식
			- 보통 딕셔너리/룰베이스 방법 사용
				- 1) List Lookup 방식
					- 간단하고 빠르지만, 리스트가 변하면 관리가 쉽지 않다.
			
				- 2) Shallow Parsing Approach
					- 대문자 단어 + Street, Avenue 같은 것은 장소라는 명확한 evidence이기 때문에, 누가봐도 고정적인 것들을 추정하는 방법 : ex) Wall Street
			
			- Model-based 방법
				- 1) MITIE
				- 2) CRF++
				- 3) CNN



### 3) Syntax Analysis(구문분석)

- 목적: 어떤 일련의 문장이 들어왔을 때 문법의 형식에 맞게 분석하는 절차
	- Ex) I(명사) love(동사) you(명사) / love와 you가 서술형파트를 구성하고 있고, I love you가 모두 합쳐져서 Sentence를 구성
	

- Parser: Input String들을 특정한 문법에 맞게 변환해주는 알고리즘
	- 두 가지 속성을 가지고 있음
		- 1. Directionality: top-down, bottom-up 방식
		- 2. Search Strategy: depth-first, breadth-first


	- Parsing Representation
		- 1) Tree Representation
		- 2) List Representation

- 구문분석도 언어의 모호성 때문에 어렵다.
	- Ex) Lexical ambiguity: Time flies like an arrow. files/like =  1.(명사, 동사), 2.(동사,전치사)로 쓰이는 경우 의미가 달라진다.
	- Ex) Structural ambiguity: Jone saw Mary in the park = 모두 동일한 품사를 가짐에도 두 가지 해석이 가능하다.
	


### 4) Other Topics in NLP

- 1) Probabilistic Language Modeling: 문장이 들어왔을 때 그 문장의 확률을 매기는 것 (문장 그 자체에 확률이 주어지는 것)
	- Machine Translation: P(**high** wind tonight) > P(**large** wind tonight)
	- Spell Correction(스펠링 체크)
	- Speech recognition: P(I saw a van) > P(eyes awe of an)
	- Summarization, QA, etc.
	
	- 확률을 계산하는 방법
		- P(w1, w2, w3, ..., wn) = P(w1)P(w2|w1)P(w3|w1, w2)... P(w_n|w1, ..., w_n-1)
		- Ex) P(its water is so transparent) = P(its)P(water|its)P(is|its water)P(so|its water is)P(transparent|its water is so)
		
	- 다만, 마지막 단어를 계산해야할 때 앞의 모든 단어가 주어져야 한다. 10개의 단어로 이루어진 문장인 경우 마지막 단어를 예측할 때 앞의 9개에 해당하는 부분의 조건부 확률을 계산해야 한다.
	- 이는 컴퓨팅 파워 및 계산이 쉽지않다.


- 2) Markov Assumption
	- Unigram Model: 각각의 단어가 독립적으로 발생했다는 것을 가정  --> 문장같이 생성되지가 않음
	- Bigram Model: 이전단어에만 영향을 받음
	
	- N-gram Model: trigrams, 4-grams, 5-grams ..
		- 그럼에도, 언어가 Long-Distance 의존성을 가지는 경우는 충분하지 않음 (ex. **The computer** when I had just put into ~~ **crashed**)
	


- 3) Neural Network-based Language Model