# 8-1. Seq2Seq Learning

- Sequence to sequence model (기계 번역 Task에서 주로 사용됨)
	- input으로 어떤 items(words, letters, 이미지, etc.)의 시퀀스를 받는 것
	- output으로 item의 another 시퀀스를 내보내는 것

	
- 핵심 아이디어: Encoder와 Decoder 두 구조가 결합되어 있음
	- Encoder : 입력 정보를 처리해서 압축해 저장하는 부분
		- 시퀀스를 특정 벡터로 압축 (context vector)

	- Decoder : Enocedr로부터 압축된 정보를 풀어서 반환(생성)하는 부분

	- 기본적으로 Encoder, Decoder는 RNN 계열로 구성되어 있음.
	- context는 하나의 벡터로 표현됨
	
- 작동 방식
	- RNN에서 시퀀스가 입력으로 들어가면 각 Time-step마다 Output벡터와 Hidden state 벡터를 출력함
	- Seq2Seq에서도 시퀀스가 Encoder에 들어가서 step을 거친 후 가장 마지막에 출력하는 Hidden state 벡터를 통해서 Output을 생성
	- Encoder를 거쳐 마지막에 출력되는 Hidden state 벡터가 Context 벡터가 됨
	
![seq2seq](https://user-images.githubusercontent.com/46666862/80800857-96ae5300-8be5-11ea-90a1-54d4e4b759eb.PNG)
	

- 문제점
	- 기존의 RNN은 Long term 시퀀스들을 반영하기가 힘듬
	- LSTM, GRU 등은 그 문제(Long term Dependency)를 해결하였지만 완벽하게 해결하진 못했음
	- Attention이라는 개념을 활용하여 문제 해결

- Attention 개념
	- Context 벡터는 Seq2Seq 구조에서 bottleneck이기 때문에 long 시퀀스를 다루는 것이 어렵다.
	- 따라서 Attention은 모델이 **각각의 input sequence들 중에서 현재 Output item이 주목해야하는 부분을 연결(가중치)**해주어 해당하는 파트의 정보를 활용할 수 있도록 하는 구조
	- Attention score를 구할 때 학습을 시키는 모델과 학습 시키지 않고 구하는 모델 존재(성능은 비슷하므로 학습하지 않는 모델이 실용적)
	
	
- Attention 매커니즘이 적용된 Seq2Seq
	- Encoder의 마지막 Hidden state만을 Decoder에 넘겨주는 것이 아니라 모든 Hidden state를 넘겨줌 (더 많은 정보를 줄 수 있음)
	- 이후 Decoding이 수행되는 과정에서 필요한 Hidden state를 취사선택해서 즉,서로 다른 가중치를 활용해서 Output을 출력
	- Decoder가 기존 step + **Extra Step**(아웃풋을 생성하기 이전에 수행)을 수행함
	- Encoder가 보내준 n개의 Hidden state를 살펴본다 (RNN 계열의 기본가정은 각 Hidden state는 각 시퀀스의 입력 단어의 정보를 가장 많이 보존하고 있을 것)
	- 즉, 2번째 Hidden state는 2번째 입력 시퀀스의 정보를 가장 많이 담고 있을 것이다.
	
![seq2seq(attention)](https://user-images.githubusercontent.com/46666862/80800862-97df8000-8be5-11ea-9c29-1972355b0ded.PNG)


- Attention 동작방식	

![attention_step](https://user-images.githubusercontent.com/46666862/80805681-a896f280-8bf3-11ea-8b45-398905d84e96.PNG)

	- Step 1) Prepare inputs 
		- 인코더를 거친 Hidden state 벡터 h1,h2,h3를 디코더에 넘겨준다
		- 디코더의 첫 input을 거치고 나온 Hidden state 벡터 h4를 준비한다
	
	- Step 2) Score each hidden state
		- h1, h2, h3와 h4를 각각 내적한다.
		- dot(h1, h4)= 13(scala),  dot(h2, h4)= 9,   dot(h3, h4)= 9   -->  [13, 9, 9] (attention vector)
	
	- Step 3) Softmax the scores
		- [13, 9, 9] 를 Softmax를 취해주어 [0.96, 0.02, 0.02]로 만들어줌
		
	- Step 4, 5) 각 h1, h2, h3 벡터에 Step3의 결과벡터를 곱하라
		- 0.96 x h1,  0.02 x h2 + 0.02 x h3
		- 4x1의 하나의 벡터가 됨
		- 이것이 Context vector임
		
	- Step 6) Context 벡터와 h4 벡터를 concat
		- context 벡터(4x1)와 h4 벡터(4x1)을 8x1의 벡터로 concat
	
	- Step 7) feedforward neural network와 연산 후 Output 벡터를 생성
		- Step6에서 concat한 8x1 벡터를 neural network에 넣어 4x1의 Output 벡터를 출력
	
	- Step 8) 출력된 Output 벡터를 다음 Step의 input으로 넣어 위의 1~7의 과정을 반복
	
![attention_step_total](https://user-images.githubusercontent.com/46666862/80806534-e4cb5280-8bf5-11ea-937e-3900b650fc14.PNG)


- Attention 결과(Heatmap)

![attention_heatmap](https://user-images.githubusercontent.com/46666862/80806764-7e92ff80-8bf6-11ea-86d3-c903fd1dc3f9.PNG)
	
	- 위 히트맵은 Je Suis etudiant가 인코더 단에서 들어가고 디코더 단에서 I가 어떤 Hidden state에 영향을 많이 받았는지를 보여준다.
	- 즉, h4 벡터와 h1,h2,h3를 각각 내적하고 Softmax를 계산한 값에 따라서 히트맵에 색이 칠해진다.
	- 위의 예시에서 I의 softmax 계산 값은 [0.96, 0.02, 0.02] 나왔으므로 I는 첫 번째인 h1에 영향을 많이 받았으므로 h1의 색이 뚜렷하게 드러난다.











