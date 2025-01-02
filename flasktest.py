from test import *

KNC_USE = joblib.load('KNC_USE.joblib')

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    new_text =  code_exterminator(text)
    new_text =  bracket_exterminator(new_text)
    new_text =  transform_dl_fct2(new_text)
    liste = [new_text]
    features_USE_liste = feature_USE_fct(liste, 1)
    predictions = KNC_USE.predict(features_USE_liste)
    predictions = predictions.tolist()[0]
    tags = ["android","google-cloud-platform","firebase","python","javascript","amazon-web-services","java","google-cloud-functions","google-cloud-firestore","firebase-realtime-database","react-native","azure","reactjs","firebase-authentication","node.js","kotlin","google-sheets","machine-learning","android-recyclerview","angular","c#","azure-active-directory","spring-boot","asp.net-core","android-fragments","kubernetes","azure-devops","python-3.x","django","typescript","aws-lambda","flutter","terraform","google-sheets-formula","django-rest-framework","android-studio","google-kubernetes-engine","google-apps-script","selenium","selenium-webdriver","google-cloud-storage","android-layout","react-navigation","aws-cloudformation","apache-kafka","selenium-chromedriver","tensorflow","azure-web-app-service","azure-pipelines","android-jetpack-compose","entity-framework-core","deep-learning","docker","apache-spark"]
    i = 0
    predictions_tags = []
    for tag in predictions:
     if tag==1:
      predictions_tags.append(tags[i])
     i+=1
    return predictions_tags