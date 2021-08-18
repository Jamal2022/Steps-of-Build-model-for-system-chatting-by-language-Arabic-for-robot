import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
#data_file = open('intents.json').read()
#intents = json.loads(data_file)
intents = {"intents": [
        {"tag": "تحية",
         "patterns": ["أهلا", "كيف حالك", "هل من أحد هناك?","يا","هلا", "مرحبا", "يوم سعيد"],
         "responses": ["مرحبا شكرا على السؤال", "سررت برؤيتك مجددا", "مرحبًا ، كيف يمكنني المساعدة؟"],
         "context": [""]
        },
        {"tag": "وداع",
         "patterns": ["وداعا", "أراك لاحقا", "مع السلامة", "لطيفة الدردشة معك ، وداعا", "حتى المرة القادمة"],
         "responses": ["أرك لاحقا!", "أتمنى لك نهارا سعيد", "وداعا! عد مرة أخرى قريبا."],
         "context": [""]
        },
        {"tag": "شكر",
         "patterns": ["شكرا", "شكرا لك", "هذا مفيد", "رائع شكرا", "شكرا لمساعدتي"],
         "responses": ["سررت بالمساعدة!", "في أي وقت!", "من دواعي سروري"],
         "context": [""]
        },
        {"tag": "لا_اجابة",
         "patterns": [],
         "responses": ["آسف ، لا أستطيع أن أفهمك", "من فضلك أعطني المزيد من المعلومات", "لست متأكدا بأني أفهم"],
         "context": [""]
        },
        {"tag": "الخيارات",
         "patterns": ["كيف يمكنك مساعدتي؟", "ما تستطيع فعله؟", "ما هي المساعدة التي تقدمها؟", "كيف يمكنك ان تكون مفيدا؟", "ما هو الدعم المقدم"],
         "responses": ["يمكنني إرشادك من خلال قائمة التفاعلات الدوائية الضارة وتتبع ضغط الدم والمستشفيات والصيدليات", "تقديم الدعم للتفاعلات الدوائية الضارة وضغط الدم والمستشفيات والصيدليات"],
         "context": [""]
        },
        {"tag": "الدواء_الضار",
         "patterns": ["كيف تتحقق من تفاعل الدواء الضار؟", "افتح وحدة الأدوية الضارة", "أعطني قائمة بالأدوية التي تسبب سلوكًا ضارًا", "ضع قائمة بجميع الأدوية المناسبة للمريض الذي يعاني من رد فعل سلبي", "ما هي الأدوية التي ليس لها رد فعل سلبي؟" ],
         "responses": ["الانتقال إلى وحدة التفاعلات الدوائية الضارة"],
         "context": [""]
        },
        {"tag": "ضغط_الدم",
         "patterns": ["افتح وحدة ضغط الدم", "مهمة متعلقة بضغط الدم", "إدخال بيانات ضغط الدم", "أريد تسجيل نتائج ضغط الدم", "إدارة بيانات ضغط الدم" ],
         "responses": ["الإنتقال إلى وحدة ضغط الدم"],
         "context": [""]
        },
        {"tag": "بحث_ضغط_الدم",
         "patterns": ["أريد البحث عن سجل نتائج ضغط الدم", "ضغط الدم للمريض", "نتيجة تحميل ضغط دم المريض", "عرض نتائج ضغط الدم للمريض", "البحث عن نتائج ضغط الدم عن طريق الهوية" ],
         "responses": ["يرجى تقديم هوية المريض", "رقم المريض؟"],
         "context": ["search_blood_pressure_by_patient_id"]
        },
        {"tag": "search_blood_pressure_by_patient_id",
         "patterns": [],
         "responses": ["تحميل نتيجة ضغط الدم للمريض"],
         "context": [""]
        },
        {"tag": "البحث_عن_صيدلية",
         "patterns": ["ابحث عن صيدلية", "ابحث عن صيدلية", "قائمة الصيدليات القريبة", "حدد موقع الصيدلية", "ابحث عن صيدلية" ],
         "responses": ["الرجاء تقديم اسم الصيدلية"],
         "context": ["search_pharmacy_by_name"]
        },
        {"tag": "search_pharmacy_by_name",
         "patterns": [],
         "responses": ["تحميل تفاصيل الصيدلية"],
         "context": [""]
        },
        {"tag": "البحث_عن_مستشفى",
         "patterns": ["ابحث عن مستشفى", "البحث عن مستشفى لنقل المريض", "أريد البحث عن بيانات المستشفى", "البحث في المستشفى عن المريض", "البحث عن تفاصيل المستشفى" ],
         "responses": ["يرجى تقديم اسم المستشفى أو الموقع"],
         "context": ["search_hospital_by_params"]
        },
        {"tag": "search_hospital_by_params",
         "patterns": [],
         "responses": ["يرجى تقديم نوع المستشفى"],
         "context": ["search_hospital_by_type"]
        },
        {"tag": "search_hospital_by_type",
         "patterns": [],
         "responses": ["تحميل تفاصيل المستشفى"],
         "context": [""]
        }
   ]
}

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
