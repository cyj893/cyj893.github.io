---
title: "Projects"
layout: archive
permalink: categories/Projects
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.Projects %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}