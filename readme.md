
# Is this a place?

## To Run

Need to use a local web server, for the ajax requests to work when loading Vue.
Express is a simple web server that doesn't take much setup.

* `npm install express`
* `node -e "require('express')().use(require('express').static(__dirname, {index:'index.html'})).listen(8181)"`
