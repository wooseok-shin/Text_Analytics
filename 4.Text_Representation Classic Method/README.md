# 3. Text Representation 1 Classic Methods

### 1) Bag of Words

- 어떤 논문의 Abstract 텍스트 데이터를 수집한다고 했을 때 각각의 Text들은 모두 길이가 다름. 이는 대부분의 머신러닝에서 사용하기 위해서는 고정된 길이로 바꾸어주어야 할 필요가 있음.
- **가변적 길이의 문서를 고정된 길이의 숫자형 벡터로 변환**하는 것, 그 중에서도 **빈도 기반의 방식**이 이 챕터의 목표

- ![4-1](https://user-images.githubusercontent.com/46666862/78241913-2051fe80-751c-11ea-874e-4aaca20ad64f.PNG)

- 위의 그림과 같이 문서1, 2, ... , n 문서마다 단어 1, 2, ... , P개까지 빈도 기반으로 나타내는 것


- Bag-of-words: 문서를 순서를 고려하지 않은 단어들의 집합으로 가정
	- Term-Document Matrix (TDM, 때로는 DTM으로 사용 = 행,열을 바꾼것)
	
	- Binary Representation: 단어가 출현했느냐 하지않았느냐 0,1로 표현
	- Frequency Representation: 단어가 몇번 출현했느냐까지 표현  (0, 1, 2, .. n)
	- 특정 Task에서는 바이너리가 성능이 더 좋을 때도 있음
	- ![4-1-2](https://user-images.githubusercontent.com/46666862/78244017-c4897480-751f-11ea-837b-a738a398f207.PNG)
	
	- 단점1. 다른 의미를 가지는 문장이나 문서가 동일한 Representation으로 표현될 가능성이 있음
		- Ex) John is quicker than Mary = Mary is quicker than John
		
	- 단점2. TDM에서 Original Text로 다시 변환 및 복원(Reconstruct)할 수가 없음, 순서를 알 수 없기 때문에
		- Ex) 오리지널(John loves Mary),  BOW(John(1), loves(1), Mary(1)) -->  Mary loves John? or John loves Mary?
		
	
	
### 2) Word Weighting

- Word Weighting: 특정한 단어가 특정한 문서에 대해서 얼마만큼 중요한지
	- 2가지 Metric
		- Term Frequency(TF): 해당하는 **특정(개별) 문서**에 해당 Term이 몇 번 등장했는지
		
		- ![tf](https://user-images.githubusercontent.com/46666862/78244080-e4209d00-751f-11ea-9613-e2d801e2fed7.PNG)
		
		- Document Frequency(DF): **전체 문서**에서 해당 Term이 몇 번 등장했는지 (한 문서에서 여러 번 등장하는 것을 세는 것이 아님, 문서에서 등장했는지 안했는지만을 계산)
			- is, can, the, of 같은 용어들은 모든 문서에서 등장할 확률이 높다. 따라서 이러한 흔한 단어들보다는 Rare한 단어들에 더 가중치를 주기 위헤서 역수를 취해준다.
			- Inverse Document Frequency(IDF): log_10(N/df) (N=Corpus 수)
			- ![IDF](https://user-images.githubusercontent.com/46666862/78244654-07981780-7521-11ea-8ce3-88646e41eef4.PNG)
			
	- 결국 특정한 단어가 특정한 뭇너에 대해서 얼만큼 중요한지를 알기 위해서 위 2가지를 결합
	- TF-IDF(w) = tf(w) x log(N/df(w))  --> 이 값이 클수록 중요
		- 하나의 타겟 문서에서 등장 빈도가 높을수록 중요성이 커진다.
		- 전체 Corpus에서 희소성이 높을수록 단어의 중요성이 커진다.
		- 즉, 하나의 문서에서 집중적으로 등장하고 전체에서는 별로 등장하지 않을 때 중요해진다.
		
		
	- 하지만, 결국 매우 고차원의 공간이고 또 매우 Sparse해진다.
	
	- TF, DF, Normalization도 변형들이 많이 있음.



### 3) N-Grams

- N-Gram in Text Mining: Text Mining에서도 N-Gram이 사용됨
	- 구텐베르크 프로젝트(N-Gram용으로 많은 코퍼스들을 모으면 번역이 잘 될 것이다. But, 뉴럴네트워크 기반이 성능이 훨씬 좋아 중단)



















