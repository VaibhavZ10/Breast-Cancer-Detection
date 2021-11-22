from flask import Flask, render_template
import GG

app=Flask(__name__,template_folder='template')

@app.route("/")
def hello():
    return render_template("index.html")


@app.route('/sub')
def submit():
    #html to py
    if request.method == "POST":
          name = request.form["username"]

    #py to html
    return render_template("sub.html", n =name)      

    if __name__ == "__main__":
        app.debug = True
        app.run()