Datasets
	:: RedV2 - target
			id2label = {0: "Tristete", 1: "Surpriza", 2: "Frica", 3: "Furie", 4: "Neutru", 5: "Incredere", 6: "Bucurie"}
			- emotii
			- ro - BERT-base-cased (dar care este varianta de multilingual-bert)
			- pe test : antrenare simpla : 67.33% F1.
			
	:: Emotions Dataset English
		20K - name	train	validation	test
			split	16000	2000	2000
		: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).  - 68.05% (pus inainte de redv2)
		https://huggingface.co/datasets/dair-ai/emotion		
			
			
	:: LaRoSeDa. The Large Romanian Sentiment Data Set                    - 95.20%
			15,000 reviews e-commerce platforms in Romania. 
			positive or negative
			
	:: Moroco: categorized web dataset cu AG News: https://paperswithcode.com/dataset/ag-news
			
	:: Romanian Categorized Web Dataset
			5000 articles
			neutral, pozitive, negative
			cultura generala - absolut orice - web articles

	:: SST2 binary sentiment classification
			movie reviews that were extracted from the Rotten Tomatoes   - 67.94% (inainte de REDv2)   - 95.28%(inainte de laroseda)
 			67K	
	
Related Work:
	:: Redv2: https://aclanthology.org/2022.lrec-1.149.pdf
				      Ham 	Acc    F1    MSE
		Ro-BERT1     0.104 0.541 0.668 26.74
		XLM-Roberta1 0.121 0.504 0.619 18.41

	++ alte articole de unde mi-au venit ideile de multi-task learning , domain adaptation, KD, data augmentation, constrastive learning.

De ce fac ?
	1. Aleg 3 task-uri:
		principal: emotii: redv2
		secundar: sentimente: laroseda
		secundar: clasificare categorii de stiri
		
	2. Pentru fiecare task fac domain adaptation cu un alt set de date din engleza, pentru ca BERT-ul pe care il voi folosi este multi-bert la baza si stie engleza.
	3. Incerc sa adaug un pas de contrastive learning (atat in functia de loss - cat si gandit ca un task auxiliar - voi lua din setul de date pe engleza un subdataset
			si il voi traduce in romana, apoi voi pune modelul sa distinga dintre propozitia in romana si cea din engleza - dar propozitiile sunt din domeniu)
				- sper ca va crea legaturi mai bune intre engleza si romana (aici am emotii ca o sa functioneaze - dar vreau sa exprimentez)
	
	4. Se presupune ca am obtinut 3 modele bune - fiecare pe task-ul sau.
	5. Urmeaza sa testez multi-task learning pe acelasi model de baza ro-bert-based-cased.
	6. Fac self-distillation, poate cu teacher annealing pentru a creste performantele.
	7. Incerc ceva data augmentation, daca mai am timp si fortez ceva ++ la scoruri. (aici sunt mai multe idei)
	

Alte schite- aceleasi idei:
	Domain Adaptation between Redv2 si EmotionsDataset
	Domain Adaptation between SST2 si RoCaWeData si LaRoSeDa
	
	Apoi intre preTraining-ul pe engleza de la Domain Adaptation introduc Contrastive Learning cu
	detectarea intre traducerea in limba romana si engleza. Apoi fine-tuning pe datseturile romanesti.
	
	Apoi, cu aceste modele finale - fac multi-task learning intre sentiment analysis si emotion analysis.
	Apoi, fac Distillation
	Apoi, fac Multi-Task Learning si Knowledge Distillation.
	Apoi, incerc si ceva Data Augmentation.
	
	
Related Work:
	- ADAN - discriminator de limba in multi-task learning - https://aclanthology.org/Q18-1039.pdf
	- LISA - cross-lingual = cu tinte de loss diferite - https://aclanthology.org/2020.eamt-1.9.pdf
	- MT Google Translate pt obtinerea dataset-ului paralel : https://aclanthology.org/P16-1133.pdf
	- CERT - Antrenare dupa antrenare fara multi-task learning : https://arxiv.org/pdf/2005.12766
	- SCL - constrastive learning - formula https://arxiv.org/pdf/2011.01403
	- 

Scriere related work:
	BERT-base-ro: (explicatii multi-bert) : https://www.semanticscholar.org/reader/8b0a0f6d1cd6f3aa9b54be45d5127bb016a98171
	Redv2: (explicatii si scor RoBERT) - sota mentionare.
	Prezentarea resturilor de seturi de date pe scurt.
	Crosslingual + constrative learning + ADAN. etc.
	Bibliografie Multi-Task Learning.
	
