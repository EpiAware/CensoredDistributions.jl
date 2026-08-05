import{_ as a,o as n,c as i,an as p}from"./chunks/framework.Dpx_PMIC.js";const d=JSON.parse('{"title":"A susceptibility-depleting renewal model","description":"","frontmatter":{},"headers":[],"relativePath":"getting-started/tutorials/renewal-susceptibility.md","filePath":"getting-started/tutorials/renewal-susceptibility.md","lastUpdated":null}'),e={name:"getting-started/tutorials/renewal-susceptibility.md"};function l(t,s,h,r,k,c){return n(),i("div",null,[...s[0]||(s[0]=[p(`<h1 id="renewal-susceptibility" tabindex="-1">A susceptibility-depleting renewal model <a class="header-anchor" href="#renewal-susceptibility" aria-label="Permalink to &quot;A susceptibility-depleting renewal model {#renewal-susceptibility}&quot;">​</a></h1><h2 id="Introduction" tabindex="-1">Introduction <a class="header-anchor" href="#Introduction" aria-label="Permalink to &quot;Introduction {#Introduction}&quot;">​</a></h2><h3 id="What-are-we-going-to-do-in-this-exercise" tabindex="-1">What are we going to do in this exercise <a class="header-anchor" href="#What-are-we-going-to-do-in-this-exercise" aria-label="Permalink to &quot;What are we going to do in this exercise {#What-are-we-going-to-do-in-this-exercise}&quot;">​</a></h3><p>We build an SIR-style renewal model with a depleting susceptible pool and follow it through to observed reported cases. Infections follow the renewal recurrence modulated by the remaining susceptible fraction; we then push the infections through an observation delay to reported counts, simulate data, and fit the model with Turing to recover the reproduction-number level and the susceptible-pool size.</p><p>Unlike the <a href="/censoreddistributions.epiaware.org/previews/PR363/getting-started/tutorials/rt-renewal-convolution#rt-renewal-convolution">Rt renewal with delay convolution</a> tutorial, which hand-rolls the renewal loop, here the recurrence is a single <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal"><code>renewal</code></a> call with a <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.susceptibility_depletion"><code>susceptibility_depletion</code></a> modulator, and the fit uses the <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal_model"><code>renewal_model</code></a> submodel so the renewal fits like the rest of the stack.</p><p>We cover: 2. A forward simulation: a generation interval, an Rt path, a susceptible pool, and an observation delay, run through <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal"><code>renewal</code></a> and <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.observe_renewal"><code>observe_renewal</code></a> to expected cases.</p><ol start="3"><li><p>A Turing fit: <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal_model"><code>renewal_model</code></a> samples the Rt path and the susceptible-pool size, and the observed counts are scored against the reported series.</p></li><li><p>Recovery: pull the posterior and check the Rt level and the pool size.</p></li></ol><h3 id="What-might-I-need-to-know-before-starting" tabindex="-1">What might I need to know before starting <a class="header-anchor" href="#What-might-I-need-to-know-before-starting" aria-label="Permalink to &quot;What might I need to know before starting {#What-might-I-need-to-know-before-starting}&quot;">​</a></h3><p>This builds on <a href="/censoreddistributions.epiaware.org/previews/PR363/getting-started/index#getting-started">Getting Started</a> and the <a href="/censoreddistributions.epiaware.org/previews/PR363/getting-started/tutorials/rt-renewal-convolution#rt-renewal-convolution">Rt renewal with delay convolution</a> tutorial, which introduces the generation interval and the <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.convolved"><code>convolved</code></a> observation layer.</p><p>The renewal recurrence <code>I[t] = R_t · (S[t]/N) · Σ_s g_s I[t-s]</code>, <code>S[t] = S[t-1] − I[t]</code>, is the <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.susceptibility_depletion"><code>susceptibility_depletion</code></a> modulator. Modulators compose with <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.combine_modulators"><code>combine_modulators</code></a>, so a transmissibility or immunity-waning term stacks on top with no change to the call.</p><h2 id="Packages-used" tabindex="-1">Packages used <a class="header-anchor" href="#Packages-used" aria-label="Permalink to &quot;Packages used {#Packages-used}&quot;">​</a></h2><p>We use Distributions for the delay families, Turing and Mooncake for the fit, FlexiChains for the chain output, and Random and Statistics for reproducibility and summaries.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CensoredDistributions</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Distributions</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Turing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Turing</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @varname</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Mooncake</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> FlexiChains</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> VNChain</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Statistics</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AutoMooncake</span></span></code></pre></div><h2 id="The-generation-interval" tabindex="-1">The generation interval <a class="header-anchor" href="#The-generation-interval" aria-label="Permalink to &quot;The generation interval {#The-generation-interval}&quot;">​</a></h2><p>The generation interval is a short discrete PMF over positive lags, built as a truncated Gamma read off in day bins with <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.interval_censored"><code>interval_censored</code></a>. We drop the zero lag (<code>lower = 1</code>) so an infection generates from the next day on, and the day-bin masses already sum to one over the positive lags.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gi_max </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 12</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gen_dist </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> interval_censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    truncated</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Gamma</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, upper </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gi_max)), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">g </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> pdf</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gen_dist, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gi_max)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>12-element Vector{Float64}:</span></span>
<span class="line"><span> 0.24328953791807156</span></span>
<span class="line"><span> 0.24667852188234635</span></span>
<span class="line"><span> 0.1909795579718374</span></span>
<span class="line"><span> 0.129674822986537</span></span>
<span class="line"><span> 0.08146631680007754</span></span>
<span class="line"><span> 0.048615330778831956</span></span>
<span class="line"><span> 0.02797041627704222</span></span>
<span class="line"><span> 0.015658985306324147</span></span>
<span class="line"><span> 0.008582753432822554</span></span>
<span class="line"><span> 0.004625330502400327</span></span>
<span class="line"><span> 0.002458426143708947</span></span>
<span class="line"><span> 0.0</span></span></code></pre></div><h2 id="The-forward-simulation" tabindex="-1">The forward simulation <a class="header-anchor" href="#The-forward-simulation" aria-label="Permalink to &quot;The forward simulation {#The-forward-simulation}&quot;">​</a></h2><p>We pick a true Rt path that rises, dips below one and recovers, a susceptible pool <code>N</code>, and a seed. <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal"><code>renewal</code></a> with <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.susceptibility_depletion"><code>susceptibility_depletion</code></a> runs the SIR-style recurrence; the susceptible fraction bends the epidemic down as the pool runs out.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n_days </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 90</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">true_Rt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fill</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fill</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">fill</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n_days </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">true_N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e4</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">I0 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 10.0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">infections </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> renewal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(true_Rt, g, I0;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    modulator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> susceptibility_depletion</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(true_N))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>90-element Vector{Float64}:</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  10.0</span></span>
<span class="line"><span>  17.784</span></span>
<span class="line"><span>  21.113799661733804</span></span>
<span class="line"><span>  25.904959900917927</span></span>
<span class="line"><span>  31.97372703132705</span></span>
<span class="line"><span>  39.448224051909605</span></span>
<span class="line"><span>  48.57847764130718</span></span>
<span class="line"><span>  59.67108749442879</span></span>
<span class="line"><span>  73.06679727343611</span></span>
<span class="line"><span>  89.12381971363055</span></span>
<span class="line"><span> 108.19333545292116</span></span>
<span class="line"><span> 130.5807805015419</span></span>
<span class="line"><span> 156.48859421483584</span></span>
<span class="line"><span> 185.92154549574587</span></span>
<span class="line"><span>  97.17055974524936</span></span>
<span class="line"><span>  93.66695657337759</span></span>
<span class="line"><span>  85.56976821061018</span></span>
<span class="line"><span>  76.59872148348276</span></span>
<span class="line"><span>  68.02843521888985</span></span>
<span class="line"><span>  60.199968667067026</span></span>
<span class="line"><span>  53.15559523366852</span></span>
<span class="line"><span>  46.85308639291352</span></span>
<span class="line"><span>  41.23009159142977</span></span>
<span class="line"><span>  36.22202002741598</span></span>
<span class="line"><span>  31.766852045512703</span></span>
<span class="line"><span>  27.80656793571924</span></span>
<span class="line"><span>  24.39028045939269</span></span>
<span class="line"><span>  21.36582667579606</span></span>
<span class="line"><span>  18.7026798957724</span></span>
<span class="line"><span>  16.36299427520906</span></span>
<span class="line"><span>  14.309919713486966</span></span>
<span class="line"><span>  12.509844446515665</span></span>
<span class="line"><span>  10.932682727210132</span></span>
<span class="line"><span>   9.551667758851531</span></span>
<span class="line"><span>   8.343054071235624</span></span>
<span class="line"><span>   7.285821493209947</span></span>
<span class="line"><span>   6.3613973860116415</span></span>
<span class="line"><span>   5.553399229570255</span></span>
<span class="line"><span>   4.847312674138956</span></span>
<span class="line"><span>   6.874519179332948</span></span>
<span class="line"><span>   6.666623092895315</span></span>
<span class="line"><span>   6.661921484102474</span></span>
<span class="line"><span>   6.719063417492149</span></span>
<span class="line"><span>   6.792386505343447</span></span>
<span class="line"><span>   6.869354375213195</span></span>
<span class="line"><span>   6.946912624362836</span></span>
<span class="line"><span>   7.024340112571862</span></span>
<span class="line"><span>   7.101395908224458</span></span>
<span class="line"><span>   7.177912920315085</span></span>
<span class="line"><span>   7.2537303794962416</span></span>
<span class="line"><span>   7.328690121655037</span></span>
<span class="line"><span>   7.3991183716261855</span></span>
<span class="line"><span>   7.468764795065754</span></span>
<span class="line"><span>   7.536801729578315</span></span>
<span class="line"><span>   7.602991915549095</span></span>
<span class="line"><span>   7.667234714271037</span></span>
<span class="line"><span>   7.729451295754443</span></span>
<span class="line"><span>   7.789564220425881</span></span>
<span class="line"><span>   7.8474962518507185</span></span>
<span class="line"><span>   7.903171399233737</span></span>
<span class="line"><span>   7.956515616431862</span></span>
<span class="line"><span>   8.007457171166687</span></span>
<span class="line"><span>   8.055926879615217</span></span>
<span class="line"><span>   8.101862932938566</span></span>
<span class="line"><span>   8.145200225191346</span></span>
<span class="line"><span>   8.18587857152653</span></span>
<span class="line"><span>   8.223841227613498</span></span>
<span class="line"><span>   8.259034807478717</span></span>
<span class="line"><span>   8.291409442658685</span></span>
<span class="line"><span>   8.320918946412151</span></span>
<span class="line"><span>   8.3475209583328</span></span>
<span class="line"><span>   8.371177073619593</span></span>
<span class="line"><span>   8.391852959933262</span></span>
<span class="line"><span>   8.409518462388991</span></span>
<span class="line"><span>   8.424147696366058</span></span>
<span class="line"><span>   8.435719121632959</span></span>
<span class="line"><span>   8.444215618911203</span></span>
<span class="line"><span>   8.449624539533875</span></span>
<span class="line"><span>   8.451937745163933</span></span></code></pre></div><h2 id="The-observation-layer" tabindex="-1">The observation layer <a class="header-anchor" href="#The-observation-layer" aria-label="Permalink to &quot;The observation layer {#The-observation-layer}&quot;">​</a></h2><p>Infections are reported after an incubation-to-onset delay, and only a fraction are ascertained. We build the delay with <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.double_interval_censored"><code>double_interval_censored</code></a> and carry the ascertainment through <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.thin"><code>thin</code></a>, then push the infections through with <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.observe_renewal"><code>observe_renewal</code></a>, the renewal-to-observation bridge.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">true_ascertainment </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.4</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">onset_delay </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> thin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    double_interval_censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Gamma</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); upper </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 20.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, interval </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    true_ascertainment)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">expected_cases </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> observe_renewal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(infections, onset_delay)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>90-element Vector{Float64}:</span></span>
<span class="line"><span>  0.3359315400213261</span></span>
<span class="line"><span>  1.398360748175655</span></span>
<span class="line"><span>  2.366517003313481</span></span>
<span class="line"><span>  3.0344445171578496</span></span>
<span class="line"><span>  3.449700739812236</span></span>
<span class="line"><span>  3.6939753100335007</span></span>
<span class="line"><span>  3.832805487873012</span></span>
<span class="line"><span>  3.9098726824510246</span></span>
<span class="line"><span>  3.9519284131634027</span></span>
<span class="line"><span>  3.9745818349011586</span></span>
<span class="line"><span>  3.986660102211083</span></span>
<span class="line"><span>  3.99304708370037</span></span>
<span class="line"><span>  4.25789077150638</span></span>
<span class="line"><span>  5.1984960366376844</span></span>
<span class="line"><span>  6.467737186993315</span></span>
<span class="line"><span>  8.023395724039693</span></span>
<span class="line"><span>  9.928995310899468</span></span>
<span class="line"><span> 12.26592823046425</span></span>
<span class="line"><span> 15.126010443470328</span></span>
<span class="line"><span> 18.613013996235587</span></span>
<span class="line"><span> 22.842933540378894</span></span>
<span class="line"><span> 27.94184012216282</span></span>
<span class="line"><span> 34.040502744403355</span></span>
<span class="line"><span> 41.26501374195183</span></span>
<span class="line"><span> 49.721512512647166</span></span>
<span class="line"><span> 55.39449722624394</span></span>
<span class="line"><span> 52.23242645670321</span></span>
<span class="line"><span> 47.07361682364166</span></span>
<span class="line"><span> 42.07916553989023</span></span>
<span class="line"><span> 37.53111355211443</span></span>
<span class="line"><span> 33.41485564306619</span></span>
<span class="line"><span> 29.691265697312065</span></span>
<span class="line"><span> 26.329002113004762</span></span>
<span class="line"><span> 23.301519924149915</span></span>
<span class="line"><span> 20.583511573267216</span></span>
<span class="line"><span> 18.149816098961008</span></span>
<span class="line"><span> 15.975509092474496</span></span>
<span class="line"><span> 14.039779653341336</span></span>
<span class="line"><span> 12.32688759223588</span></span>
<span class="line"><span> 10.81447426443917</span></span>
<span class="line"><span>  9.480717688596952</span></span>
<span class="line"><span>  8.305944997506785</span></span>
<span class="line"><span>  7.272436331783005</span></span>
<span class="line"><span>  6.364199137107103</span></span>
<span class="line"><span>  5.56682540909526</span></span>
<span class="line"><span>  4.867583497825263</span></span>
<span class="line"><span>  4.254668005337277</span></span>
<span class="line"><span>  3.717796397043437</span></span>
<span class="line"><span>  3.2478129391617507</span></span>
<span class="line"><span>  2.8365929959964626</span></span>
<span class="line"><span>  2.5657647688706</span></span>
<span class="line"><span>  2.543360365109276</span></span>
<span class="line"><span>  2.575336353051174</span></span>
<span class="line"><span>  2.6090648575255333</span></span>
<span class="line"><span>  2.640531456855509</span></span>
<span class="line"><span>  2.6710786374779105</span></span>
<span class="line"><span>  2.701543342932339</span></span>
<span class="line"><span>  2.732154859444452</span></span>
<span class="line"><span>  2.7628925636100585</span></span>
<span class="line"><span>  2.793667499694419</span></span>
<span class="line"><span>  2.824383828856673</span></span>
<span class="line"><span>  2.8549535589558817</span></span>
<span class="line"><span>  2.885179328700008</span></span>
<span class="line"><span>  2.9147460645763736</span></span>
<span class="line"><span>  2.9436373511993943</span></span>
<span class="line"><span>  2.971856631801783</span></span>
<span class="line"><span>  2.999382750770007</span></span>
<span class="line"><span>  3.0261843901421077</span></span>
<span class="line"><span>  3.052228131237741</span></span>
<span class="line"><span>  3.077480841687969</span></span>
<span class="line"><span>  3.1019057175036693</span></span>
<span class="line"><span>  3.125477192443608</span></span>
<span class="line"><span>  3.148163456980434</span></span>
<span class="line"><span>  3.1699342690743424</span></span>
<span class="line"><span>  3.190760663085876</span></span>
<span class="line"><span>  3.2106149916037277</span></span>
<span class="line"><span>  3.229470523100092</span></span>
<span class="line"><span>  3.2473015629658954</span></span>
<span class="line"><span>  3.2640835852777794</span></span>
<span class="line"><span>  3.279793315275925</span></span>
<span class="line"><span>  3.2944087956886463</span></span>
<span class="line"><span>  3.307909448977951</span></span>
<span class="line"><span>  3.3202761429357768</span></span>
<span class="line"><span>  3.331491231951529</span></span>
<span class="line"><span>  3.3415386220198764</span></span>
<span class="line"><span>  3.350403816888095</span></span>
<span class="line"><span>  3.3580739614244304</span></span>
<span class="line"><span>  3.3645378805507584</span></span>
<span class="line"><span>  3.3697861139086442</span></span>
<span class="line"><span>  3.373810945207282</span></span></code></pre></div><p>The expected counts are Poisson means; we draw observed counts as the data the fit will see.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> MersenneTwister</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20260626</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">cases_obs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Poisson</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(expected_cases </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e-6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>90-element Vector{Int64}:</span></span>
<span class="line"><span>  0</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  2</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  9</span></span>
<span class="line"><span> 14</span></span>
<span class="line"><span> 10</span></span>
<span class="line"><span> 14</span></span>
<span class="line"><span> 25</span></span>
<span class="line"><span> 33</span></span>
<span class="line"><span> 36</span></span>
<span class="line"><span> 38</span></span>
<span class="line"><span> 60</span></span>
<span class="line"><span> 55</span></span>
<span class="line"><span> 58</span></span>
<span class="line"><span> 55</span></span>
<span class="line"><span> 47</span></span>
<span class="line"><span> 38</span></span>
<span class="line"><span> 29</span></span>
<span class="line"><span> 31</span></span>
<span class="line"><span> 32</span></span>
<span class="line"><span> 27</span></span>
<span class="line"><span> 26</span></span>
<span class="line"><span> 12</span></span>
<span class="line"><span> 16</span></span>
<span class="line"><span> 19</span></span>
<span class="line"><span> 10</span></span>
<span class="line"><span>  9</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  7</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  0</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  6</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span>  2</span></span>
<span class="line"><span>  1</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  2</span></span>
<span class="line"><span>  1</span></span>
<span class="line"><span>  1</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  0</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span> 10</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  2</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  9</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  3</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  4</span></span>
<span class="line"><span>  5</span></span>
<span class="line"><span>  0</span></span>
<span class="line"><span>  4</span></span></code></pre></div><h2 id="The-Turing-fit" tabindex="-1">The Turing fit <a class="header-anchor" href="#The-Turing-fit" aria-label="Permalink to &quot;The Turing fit {#The-Turing-fit}&quot;">​</a></h2><p><a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal_model"><code>renewal_model</code></a> takes the Rt path and the modulator priors, runs the renewal, and returns the infections. We report them through the same observation delay and score the observed counts. The <code>make_modulator</code> closure maps the sampled <code>logN</code> to a <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.susceptibility_depletion"><code>susceptibility_depletion</code></a> modulator, so the fit and the simulation share the recurrence. Here we hold Rt at its known path (passed as a plain vector, so no Rt parameters enter the chain) and estimate the susceptible-pool size, the quantity the depletion adds over the bare renewal. The pool prior is bounded: at very large <code>N</code> the susceptible fraction saturates near one and the depletion gradient vanishes, so a bounded prior keeps the sampler on the identifiable slope. The <a href="/censoreddistributions.epiaware.org/previews/PR363/getting-started/tutorials/rt-renewal-convolution#rt-renewal-convolution">Rt renewal with delay convolution</a> tutorial covers estimating the Rt path itself.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mod_priors </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> truncated</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0e4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.0e3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 4.0e3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    upper </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2.5e4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">make_mod </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> p </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> susceptibility_depletion</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(p</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> renewal_fit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(g, I0, Rt, mod_priors, make_mod, delay, cases)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    infections </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> to_submodel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">renewal_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(g, I0, Rt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        modulator_priors </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mod_priors, make_modulator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> make_mod))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    expected </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> observe_renewal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(infections, delay)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cases </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> product_distribution</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Poisson</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(expected </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e-6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> infections</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> renewal_fit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(g, I0, true_Rt, mod_priors, make_mod, onset_delay,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cases_obs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">chain </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), model,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    NUTS</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.95</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; adtype </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoMooncake</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; config </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    150</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; chain_type </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> VNChain, progress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Info: Found initial step size</span></span>
<span class="line"><span>└   ϵ = 0.2</span></span></code></pre></div><h2 id="Recovery" tabindex="-1">Recovery <a class="header-anchor" href="#Recovery" aria-label="Permalink to &quot;Recovery {#Recovery}&quot;">​</a></h2><p>We read the posterior susceptible-pool size back from the chain. The pool size is namespaced under the <code>renewal_model</code> submodel prefix, and the posterior mean recovers the data-generating value.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">recovered_N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(chain[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@varname</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(infections</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x)])</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>10447.362058252802</span></span></code></pre></div><p>The recovered pool size sits near the true <code>N</code>, and the renewal fits through the same <a href="/censoreddistributions.epiaware.org/previews/PR363/lib/public#CensoredDistributions.renewal"><code>renewal</code></a> recurrence the simulation used — the susceptibility-depletion story carried end to end from incidence to reported cases.</p><p>This is the population-renewal side of the individual-path correspondence the recurrent multi-state work (#545) and the convolve-loop population view (#759) build out; those are separate from this scalar-incidence renewal step.</p>`,38)])])}const g=a(e,[["render",l]]);export{d as __pageData,g as default};
