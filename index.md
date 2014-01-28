---
layout: page
title: on exploratory data analysis & visualization
tagline: 
---
{% include JB/setup %}





## Who am I?

I am [Batman](http://mayankmisra.com/read-me/)

Connect:
[LinkedIn](http://linkedin.com/in/mayankmisra) 
[Twitter] (http://twitter.com/mayankmisra)

## Working on

Getting this site up and running
1 Publishing thoughts on interesting graphs I come across
1 Ramping up on python and R as pre reqs for [EDAV] (http://malecki.github.io/edav/agenda.html)
1 Trying to keep up with Machine Learning [assignments] (http://www1.ccls.columbia.edu/~ansaf/4721/assignements.html)

## Recent Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>




