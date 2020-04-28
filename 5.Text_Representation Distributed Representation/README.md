# 5. Text Representation 2 Distributed Representation

### 1) Word-level: NNLM (Neural Network Language Model)

- 분산표현: 단어를 특정 공간상의 연속형 수치형 벡터로 치환(맵핑)하는데 만약 단어들의 의미가 유사하다면 공간상에서 가까운 위치의 두 점으로 맵핑되도록 하는 것.

- 기존에 BOW 기반의 One-Hot vector는 단어간의 similarity가 0이다. (각 원핫벡터끼리의 내적이 항상 0)

- words --> R^n 으로 줄이는 것  ( n < 단어의 수)

- 목적: 원핫인코딩의 Cusre of demensionality를 해결하기 위하여
	- 1. 각 단어를 Distributed Word feature vector로 표현할 수 있음(실수 공간)
	- 2. 단어들이 함께 나올 확률을 계산할 수 있음


- 동작원리 (이전 BOW의 N-gram 모델처럼 이전의 N개의 단어를 가지고 동작한다, FeedForward 네트워크)
	- 수식: ![NNLM 수식](https://user-images.githubusercontent.com/46666862/80338748-16f84f80-8898-11ea-84c6-e5c8dd83b525.gif)
	- 수식설명: f라는 함수를 g와 C라는 두 개로 Decompose, f의 w는 단어이고 C의 w는 단어가 임베딩된 벡터
	- i는 인덱스, 다음 단어가 될만한 가능성이 있는 단어
	
	- 1. C = Lookup Table(V, m)
	- 2. g = 앞의 몇개의 단어가 등장했을 때 특정 마지막 단어가 등장할 확률
		- Ex) (준다 | 너에게, 나의 입술을, 처음으로)의 확률은 높게, g(지운다 | 너에게, 나의 입술을, 처음으로)의 확률은 낮도록
		- '처음으로, 나의 입술을, 너에게' 임베딩된 세 단어(벡터)가 입력으로 들어가고 이후에 등장할 단어의 확률이 최대가 되도록 하는 Neural Network 모델을 만드는 것.
		- Language Model은 다음 단어가 무엇인지 알 수 있으므로 다음 단어의 확률이 높게 나타나도록 w(Parameter)를 학습함
		
	- Output은 단어의 수(|v|)만큼의 차원 (Softmax로 각 단어가 나올 확률을 계산)



### 2) Word-level: Word2Vec

- CBOW, Skip-gram 두 가지 방식이 존재
	- CBOW: w(t-2), w(t-1), w(t+1), w(t+2)를 가지고 w(t)를 예측하는 것
	- Skip-gram: w(t)를 가지고 w(t-2), w(t-1), w(t+1), w(t+2)를 예측하는 것
	
- 정보량은 CBOW가 많아 보이지만 실제로 성능은 Skip-gram이 더 좋다.
	- Gradient Flow 관점에서 설명을 해보자.
		- CBOW는 1개의 output w(t)가 gradient를 4개의 단어에 나누어주지만, Skip-gram은 4개의 output이 1개의 w(t)에 gradient를 주기 때문에 성능이 더 좋다고 볼 수 있다.
		- 즉, 4번의 업데이트(skip-gram) > 1번의 업데이트(CBOW)
		
		
- **Skip-gram**
	- **Activation Function이 존재하지 않는다.**
	- 따라서, Input Layer --> Hidden Layer --> Output Layer가 모두 Linear한 구조로 이루어진다.
	- Objective Function: ![Skipgram_목적함수](https://user-images.githubusercontent.com/46666862/80341369-4493c780-889d-11ea-846f-930be5712ea1.PNG)
		- 어떤 Center word가 주어졌을 때, 위의 식을 Maximize하는 것
	- 예를들어 'The mighty **knight** Lancelot fought.' 라는 문장이 있을 때, knight이 중심 단어
	
	- Skipgram 확률 공식
		- ![skipgram_prob](https://user-images.githubusercontent.com/46666862/80341836-182c7b00-889e-11ea-9b85-002233c138ad.PNG)
		- 따라서, 주어진 w(t)의 context word의 확률은 위와 같이 Xw와 Vc의 내적으로 계산됨
		
	
	- ![skipgram_prob2](https://user-images.githubusercontent.com/46666862/80348504-d8b75c00-88a8-11ea-8ad0-55c6fe71230d.PNG)
		- 공식을 간단히 하기 위해 p(w_t+j | w_t) 대신 위와같이 p(o|c)로 표현하기도 한다.
		- Input과 Hidden Layer 사이의 Weight matrix를 W라 하고 Hidden과 Output Layer 사이의 matrix를 W'이라 한다.
		- 따라서 v_c는 특정 Center word의 워드벡터로서 W에서 값을 가져오고, 
		- u_o는 output word이고, W'에서 값을 가져오고 그 후에 v_c와 내적을 한다. (내적하면 스칼라 값 하나가 나옴)
		- 그러나, 보통 계산의 효율상 W' = W'T(W의 Transpose)로 두고 계산한다.

	- Gradient를 통해 Learning 하는 과정
		- ![skipgramGD](https://user-images.githubusercontent.com/46666862/80349745-a4dd3600-88aa-11ea-83dd-99ebca634e97.PNG)
		- J(theta)는 최대화시켜야할 목적함수
		- p(o|c)는 center word가 주어졌을 때의 확률 (하나의 스칼라 값)
		- 이후, 목적함수인 log p(o|c)를 v_c(특정 center word의 워드벡터)로 편미분을 함.
		- 마지막에 두 개의 식 중 왼쪽을 A, 오른쪽을 B라고 두고 따로따로 계산해보자.
		- ![skipgram_AB](https://user-images.githubusercontent.com/46666862/80350452-b410b380-88ab-11ea-9cbc-e64a4d6f760b.PNG)
		- ![skipgram_3](https://user-images.githubusercontent.com/46666862/80350530-d3a7dc00-88ab-11ea-8442-961997b394c9.PNG)
		- 위의 순서로 계산이 된다. (Maximize이므로 Gradient Ascent)

	- 학습 Strategy
		- 1. 한 번에 여러 Output의 Gradient를 계산하는 것이나, 각각 Output의 Gradient를 계산해서 더하는 것이나 동일하다.
			- 그러므로, 간단하게 하나씩 계산을 해라.
			
		- 2. Weight의 사이즈는 2xVxN (매우 크다), 계산을 심플하게 하기 위한 전략
			- (1) 매우 자주 등장하는 단어구는 하나의 단어로 학습을 해라 (Machine Learning, Artificial Intelligence)
			- (2) 빈번하게 등장하는 단어를 Subsampling 해라 (관사와 같이 매우 많이 등장하는 단어는 training 시에 특정 비율만큼 제거후 학습, 공식이 있음)

		- 3. Negative Sampling
			- 기존의 p(o|c)를 계산하려면 Softmax처럼 center word 이외의 모든 단어에 대해서 내적값을 구해야 한다.
			- 이렇게 되면 단어의 수(v) 만큼 반복해서 내적값을 구해야하는데 이는 계산효율이 매우 좋지 않으므로 window size 외에 있는 Negative(정답이 아닌) 단어 중 몇 개(k)만 sampling해서 확률을 구해준다.
			



### 3) Word-level: Glove

- Word2Vec의 문제점을 제기 (관사와 같이 너무 많이 등장하는 단어들을 학습하는데 많은 시간을 사용함)
	- the와 같은 단어들이 매우 많이 등장하므로 Weight를 Update할 때 the가 너무 자주 사용되어 불균형이 존재한다.
	- Word2Vec에 sub-sampling같은 기법이 존재하지만 그조차도 충분하지 않다는 논리로 전개
	
- Matrix factorization 기반 방법
	- LSA는 말뭉치 전체의 통계적인 정보를 모두 활용하지만 단어/문서 간의 유사도 측정이 어렵다
	- Word2Vec은 사용자가 지정한 윈도우(주변 몇 단어) 내에서만 학습/분석이 이루어지므로 코퍼스 전체의 co-occurrence는 반영되기 어렵다는 단점이 있다.
	- 따라서, 두 단어벡터의 내적이 코퍼스 전체에서의 동시 등장확률 로그값을 목적함수로 정의
	- **임베딩된 단어벡터 간 유사도 측정을 수월하게 하면서도 말뭉치 전체의 통계 정보를 좀 더 잘 반영해보는 것이 핵심**


- Notations
	- co-occurrence matrix (size: V x V)
	- X_ij: word i와 j가 함께 등장한 빈도
	- X_i : word i가 corpus에서 등장한 횟수
	- P_ij : P(j|i) = X_ij / **X_i**  (분모가 X_i)
	
- Motivation
	- 특정 k라는 단어가 i, j 두 단어 중 어떤 단어와 더 연관성이 클까에 대해서 생각을 해보면
	- Ratio P_ik / P_jk 값이 크면 k는 word i와 관련이 있고 값이 작으면 k는 word j와 관련이 더 많다.
	- 만약 Ratio 값이 1에 가까우면 word i와 j는 k와 둘 다 연관이 있거나 둘 다 연관이 없는 것이다.
	

- Formulation (**관계를 표현하기 위해 식을 변형**)
	- 세 단어간의 관계를 표현하기 위한 Function : F(w_i, w_j, w_k) = P_ik / P_jk
	- 두 단어간의 관계를 표현하기 위해 Subtraction : F(w_i - w_j, w_k) = P_ik / P_jk
	- 두 단어간의 관계(차이)와 Context word 사이의 Link를 만들어주기 위한 것: F{(w_i - w_j)^T w_k) (내적으로 스칼라 값) = P_ik / P_jk
	
- 위의 Formulation는 다음과 같은 조건을 만족해야 함
	- 1. w_i와 w_k를 서로 바꾸어도 식이 같은 값을 반환해야 한다. (w_i <-> w_k)
		- center word(w_k)는 얼마든지 w_i나 w_j가 될 수 있기 때문이다. (apple과 banana의 관계나 banana와 apple의 관계는 같아야 함)
	
	- 2. X(co-occurrence matrix, VxV)는 대칭행렬이므로 F는 이러한 성질을 포함 (X <-> X^T)
	- 3. homomorphism 조건을 만족해야 한다. (벡터 공간의 성질을 보존, 즉 맵핑이 되는 함수여야 한다.)
		- 여기서는 3번 식 F{(w_i - w_j)^T w_k) 을 P_ik / P_jk로 맵핑할 때,
		- w_i - w_j가 w_j - w_i로 바뀌면 (P_ik / P_jk)가 (P_jk / P_ik)로 역수로 바뀌어야 한다.
		- a란 식이있으면 덧셈의 항등원은 -a 이고 곱셈의 항등원은 1/a가 된다.
		- 즉, F라는 함수에서는 덧셈이 곱셈으로 바뀌는 (R, +) to (R>0, x) 준동형 사상을 필요로 한다.
		- f(a+b) = f(a) x f(b)를 만족하는 함수 (지수함수가 대표적)
	
- Solution
	- F(x) = exp(x)를 사용해 공식을 유도
	- ![glove_solution](https://user-images.githubusercontent.com/46666862/80455685-31e5c500-8967-11ea-81f9-705b66817261.PNG)
	- 가장 마지막 식을 보면 w_i(T)와 w_k의 내적(i와 k의 관계)과 상수항 텀이 log X_ik(co-occurrence matrix에서 구할 수 있는 i와 k가 동시에 등장하는 빈도)인 것을 알 수 있다.
	
	
- Objective Function
	- ![glove_objective](https://user-images.githubusercontent.com/46666862/80455949-aa4c8600-8967-11ea-92be-ba8c5d98d7c3.PNG)
	- 빨간색 부분은 우리가 알고있는 관측값, 나머지 부분은 학습을 통해 찾아야하는 부분
	- 그리고 두 번째 식에는 f(X_ij)가 붙어있는데, 이는 관사와 같은 고 빈도 단어에 대해서는 학습에 대한 가중치를 줄여주는 부분
	- f(x)에 관한 식은 아래와 같다.
	- ![glove_f(x)](https://user-images.githubusercontent.com/46666862/80456361-6b6b0000-8968-11ea-8337-814bd884d0c2.PNG)
	
	
	
	
	
	





