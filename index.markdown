---
layout: single
author_profile: true
---

👋 **Hi, I’m Simone Piccinini**  

I’m a 20-year-old **Software Engineering student** at the University of Padua, Italy.
I have a deep passion for **science** and **mathematics**, and I’m fascinated by how these fields can be applied to **machine learning** to tackle real-world problems. 
This curiosity fuels my **analytical thinking** and **problem-solving skills**, and I’m always eager to learn new things 📚.  

🎵 **A bit more about me:**  
- I’ve been playing **drums** since I was young 🥁  
- I enjoy playing **tennis 🎾**  

## My blog:

<ul>
{% for post in site.posts %}
  <li><a href="{{ post.url }}">{{ post.title }}</a> – <small>{{ post.date | date: "%b %-d, %Y" }}</small></li>
{% endfor %}
</ul>