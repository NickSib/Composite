{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2895637d-159b-4bd4-b421-e4356a770f44",
   "metadata": {},
   "source": [
    "##### Flask-приложение, предсказывающее значения модуля упругости при растяжении"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ebc746e9-3e7b-49a2-958e-94593661c100",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import click\n",
    "import pickle\n",
    "from flask import Flask, render_template, request\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = pickle.load(open('ln_model.pkl', 'rb'))\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    int_features = [float(x) for x in request.form.values()]\n",
    "    final_features = [np.array(int_features)]\n",
    "    prediction = model.predict(final_features)\n",
    "    output = round(prediction[0], 2) \n",
    "    return render_template('index.html', prediction_text='Модуль упругости при растяжении, ГПа :{}'.format(output))\n",
    "\n",
    "app.run()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af8d69f3-b3cb-41ae-8a01-02cd3c583307",
   "metadata": {},
   "source": [
    "##### Проверка корректности предсказания"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa0ba714-d0ac-436e-8588-dc44b42e05f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "model = pickle.load(open('ln_model.pkl', 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bed94a64-8273-4086-862a-5d1f5cc0dc35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([76.84437407])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test = np.array([5, 3, 5, 7, 4, 4, 4, 43, 4, 11, 4])\n",
    "test = test.reshape(1, -1)\n",
    "test_pred = model.predict(test)\n",
    "test_pred"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
