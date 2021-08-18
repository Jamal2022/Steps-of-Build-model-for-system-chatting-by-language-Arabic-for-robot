import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, uic

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
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
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def send():
     
    dlg.textEdit_2.append("أنت : "+str(dlg.textEdit.toPlainText()) +"\n"+"الرد الآلي : "+str(chatbot_response(dlg.textEdit.toPlainText())))
    dlg.textEdit.clear() 


app = QtWidgets.QApplication([])
dlg = uic.loadUi("Chat.ui")
dlg.show()

dlg.pushButton.clicked.connect(send)
app.exit(app.exec_())
