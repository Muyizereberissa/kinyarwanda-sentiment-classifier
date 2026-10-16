# Error analysis (test set samples)

## Correct predictions

- **neutral** (0.54): @user @user @user Kubeshya ntabwo Ari icyaha uretse iyo urukiko rubyemeje
- **neutral** (0.53): @user Itegeko ry'umuryango rivuga ko n'umugore yaba umutwe w'urugo.
- **negative** (0.57): Abafollowers ba byimbisha intugu ba @user @user nta shati zibakwira babaze wumve??😅😂 <URL>
- **neutral** (0.51): @user @user Aha niho bita ku kirenge ra?
- **negative** (0.70): Hari abantu bahisemo brain kuyikoresha nk’amasogisi mw’iyi minsi😂😂😂
- **positive** (0.73): @user @user @user @user Bugesera Courage abana bahugire mu myidagaduro hari ubuzima bubafasha kwitwara neza Kandi bikanabarinda uburwayi
- **negative** (0.54): Umugore mwiza ntasohora umutwe mwishuka iyo umugabo asuze. <URL>
- **negative** (0.44): Njye Nasura hariya hantu yezu yakoreye i Gitangaza gikomeye cyo guhindura amazi divayi. Ingwasi y'icyaka ntitwari kuzayikira😂 <URL>

## Misclassifications

- True **negative** → Pred **neutral** (0.57): @user @user @user Injiji ni wowe. Ikibazo cya mbere n'uko mutimenya!
- True **negative** → Pred **positive** (0.58): @user @user Ntabe ari uhutazwa rwose dore niyemeje kwivanga mubyawe🤣
- True **positive** → Pred **neutral** (0.70): @user na @user muze mumpe Congs ndyame! 😂
- True **positive** → Pred **negative** (0.58): @user 😂😂😂poleni gisa 👊 nta storm idashira ngo hacye🤗
- True **positive** → Pred **neutral** (0.53): 🤗 Ndagukunda ujya unkomeza mu magambo umbwira🙂😙 <URL>
- True **positive** → Pred **neutral** (0.48): @user @user @user @user @user @user Bata umwanya kandi bari mu kazi muvandi🤣🤣 nibyiza kuko n'umutoza arimo baramufasha mu bitekerezo
- True **positive** → Pred **neutral** (0.69): Niba usobanukiwe ibimenyetso ubona ku bikarito? Dore ubusobanuro ngubu! #RwOT #Rwanda #Rwandan #menya <URL>
- True **neutral** → Pred **positive** (0.75): Muri iki gitondo, Nsuhuje Abantu mwese muri muri Repubulika Yigenga Ya Kivu. Abaturanyi banyu #RDCongo, #Tanzania #Burundi #Rwanda #Ouganda turabatashya cyane. <URL>
- True **positive** → Pred **negative** (0.56): @user @user @user Harya nangahe ngo nyohereze
- True **neutral** → Pred **negative** (0.76): @user @user Nuko agenda yitura hasi ntiyamenyereye imisozi gusa banza uruta Jackson
- True **negative** → Pred **neutral** (0.44): @user @user @user Ariko Mana ubwose ushatse kuvuga iki? Mubyukuri Gusa ndumiwe cyane.
- True **neutral** → Pred **negative** (0.44): Ariko inkingo zose twafashe kuva tukiri impinja nizitubana nyinci mu mubiri bizagenda gute? #rwot #COVID19