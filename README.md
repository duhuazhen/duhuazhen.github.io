<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>


formula1:
$$n = x$$

formula2: $$n \neq x$$

formula3:
\(m = y\)

formula4: \[m \neq y\]

# DuHuazhen Blog

 ###[View Live DuHuazhen Blog &rarr;](https://duhuazhen.github.io)  <br>


## 公式的编写      

### 方法一  
 
到Latex在线编辑网站  
http://www.codecogs.com/latex/eqneditor.php  编写 

<img src="http://latex.codecogs.com/gif.latex?x_{a}^{b}=\sqrt{x}&plus;_{a}^{b}\textrm{c}" title="x_{a}^{b}=\sqrt{x}+_{a}^{b}\textrm{c}" />

### 方法二：使用MathJax引擎
大家都看过Stackoverflow上的公式吧，漂亮，其生成的不是图片。这就要用到MathJax引擎，在Markdown中添加MathJax引擎也很简单:  

```javascript
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
```
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$


<!-- mathjax config similar to math.stackexchange -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
        inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    messageStyle: "none",
    "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
