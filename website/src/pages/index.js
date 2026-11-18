import Link from '@docusaurus/Link';
import {translate} from '@docusaurus/Translate';
import Layout from '@theme/Layout';

import styles from './index.module.css';

// One card per chapter in ../docs. The URL is the file name without number prefix and ".md".
// translate() returns the text in the current language; the German texts are in i18n/de/code.json.
// The ids must be written out literally so `npm run write-translations` can find them.
function getChapters() {
  return [
    {
      path: '/docs/quickstart',
      title: translate({id: 'home.chapter.quickstart.title', message: 'Quickstart'}),
      text: translate({id: 'home.chapter.quickstart.text', message: 'From zero to a first RAG answer in about ten minutes.'}),
    },
    {
      path: '/docs/setup',
      title: translate({id: 'home.chapter.setup.title', message: '1. Setup & Configuration'}),
      text: translate({id: 'home.chapter.setup.text', message: 'uv, Docker, .env and how settings reach every module.'}),
    },
    {
      path: '/docs/ingestion',
      title: translate({id: 'home.chapter.ingestion.title', message: '2. Document Ingestion'}),
      text: translate({id: 'home.chapter.ingestion.text', message: 'Load, clean, split, embed and store - and how pgvector stores vectors.'}),
    },
    {
      path: '/docs/retrieval',
      title: translate({id: 'home.chapter.retrieval.title', message: '3. Retrieval Strategies'}),
      text: translate({id: 'home.chapter.retrieval.text', message: 'Similarity, MMR, score threshold, metadata filter and hybrid search.'}),
    },
    {
      path: '/docs/langchain_rag',
      title: translate({id: 'home.chapter.langchain.title', message: '4. LangChain RAG Chain'}),
      text: translate({id: 'home.chapter.langchain.text', message: 'A fixed pipeline built with LCEL: retriever, prompt, LLM.'}),
    },
    {
      path: '/docs/langgraph_rag',
      title: translate({id: 'home.chapter.langgraph.title', message: '5. LangGraph RAG Workflow'}),
      text: translate({id: 'home.chapter.langgraph.text', message: 'The same pipeline as a graph: state, nodes and edges.'}),
    },
    {
      path: '/docs/agentic_rag',
      title: translate({id: 'home.chapter.agentic.title', message: '6. Agentic RAG'}),
      text: translate({id: 'home.chapter.agentic.text', message: 'A graph that decides, grades documents, rewrites queries and retries.'}),
    },
    {
      path: '/docs/comparison',
      title: translate({id: 'home.chapter.comparison.title', message: '7. Comparison'}),
      text: translate({id: 'home.chapter.comparison.text', message: 'LangChain vs. LangGraph, deterministic vs. agentic RAG.'}),
    },
    {
      path: '/docs/pdf_chunking',
      title: translate({id: 'home.chapter.pdf.title', message: '8. PDF Chunking by Font Size'}),
      text: translate({id: 'home.chapter.pdf.text', message: 'Split PDFs into chapters and sections with pdfplumber.'}),
    },
    {
      path: '/docs/testing',
      title: translate({id: 'home.chapter.testing.title', message: '9. Testing'}),
      text: translate({id: 'home.chapter.testing.text', message: 'What the tests check, how to read their output, and how to write your own.'}),
    },
    {
      path: '/docs/llm_as_judge',
      title: translate({id: 'home.chapter.judge.title', message: '10. LLM-as-a-Judge'}),
      text: translate({id: 'home.chapter.judge.text', message: 'Grade RAG answers with LLM judges: correctness, relevance, groundedness, retrieval relevance.'}),
    },
    {
      path: '/docs/retrieval_metrics',
      title: translate({id: 'home.chapter.metrics.title', message: '11. Retrieval Metrics: Hit@k and MRR'}),
      text: translate({id: 'home.chapter.metrics.text', message: 'Label random chunks with LLM-generated questions and measure how high each strategy ranks them.'}),
    },
    {
      path: '/docs/mlflow',
      title: translate({id: 'home.chapter.mlflow.title', message: '12. MLflow'}),
      text: translate({id: 'home.chapter.mlflow.text', message: 'Trace every LangChain/LangGraph run and compare evaluation runs - MLflow server in Docker.'}),
    },
  ];
}

function ChapterCard({path, title, text}) {
  return (
    <Link to={path} className={styles.card}>
      <h3>{title}</h3>
      <p>{text}</p>
    </Link>
  );
}

export default function Home() {
  const tagline = translate({
    id: 'home.tagline',
    message: 'Learn Retrieval-Augmented Generation step by step - with LangChain, LangGraph and PostgreSQL/pgvector.',
  });
  return (
    <Layout title={translate({id: 'home.pageTitle', message: 'Home'})} description={tagline}>
      <header className={styles.hero}>
        <h1>RAG Learning Lab</h1>
        <p>{tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/quickstart">
            {translate({id: 'home.quickstartButton', message: 'Quickstart'})}
          </Link>
          <Link className="button button--secondary button--lg" to="/docs/setup">
            {translate({id: 'home.startButton', message: 'Start with chapter 1'})}
          </Link>
        </div>
      </header>
      <main className={styles.chapters}>
        {getChapters().map((chapter) => (
          <ChapterCard key={chapter.path} {...chapter} />
        ))}
      </main>
    </Layout>
  );
}
