# AskMyDoc — From RAG Demo to Reliable AI Product

I built AskMyDoc because I wanted to understand how modern AI applications actually work beneath the surface.

At the time, PDF chat applications were everywhere. Rather than trying to invent a novel AI product, I deliberately chose a problem that was already well understood. My goal wasn't to build a startup idea. It was to learn the engineering behind retrieval-augmented generation (RAG) systems and gain hands-on experience building one from scratch.

The first version started as a simple experiment.

I wanted to understand the entire flow of a RAG application: extracting text from PDFs, splitting documents into chunks, generating embeddings, storing vectors, retrieving relevant passages, and using those passages to generate grounded answers.

The MVP successfully allowed users to upload a PDF and ask questions about its contents. Under the hood it used FastAPI, Next.js, LangChain, OpenAI models, and ChromaDB. From a technical perspective it worked, and more importantly, it taught me how the core pieces of a RAG pipeline fit together.

Once the fundamentals were working, I focused on polishing the experience. I redesigned the interface, introduced a cleaner upload-first workflow, added dark mode, improved documentation, and rebranded the project from Context-PDF to AskMyDoc. I also cleaned up configuration management by moving model selection into environment variables instead of hardcoded values.

By the end of V1, AskMyDoc had become a polished single-document RAG application.

However, I started noticing a problem.

The AI worked, but the application itself felt temporary.

There was no authentication. Documents weren't tied to users. Conversations disappeared whenever the session ended. Nothing persisted beyond the current browser session. The project demonstrated retrieval and question answering, but it didn't behave like a product that people could actually use over time.

That realization became the motivation behind V2.

The focus of V2 wasn't better retrieval or more advanced AI techniques. Instead, it was about productizing the application.

I introduced authentication through Google OAuth and credential-based login. Documents became associated with individual users. Conversations and messages were stored permanently so users could leave and return later without losing their work. I added conversation history, document management, search functionality, user profiles, and a sidebar-driven workspace experience inspired by modern AI products.

The question I was trying to answer changed.

V1 asked:

> Can I build a RAG application?

V2 asked:

> Can I build an AI application that behaves like a real product?

By the time V2 was complete, users could upload documents, manage them, revisit previous conversations, and continue working where they left off. The system had evolved beyond a technical demo and started feeling like software people could genuinely use.

But another lesson emerged.

As the application grew, I realized that correctness and reliability were becoming bigger challenges than retrieval itself.

The frontend and backend had to agree on increasingly complex request and response structures. User ownership needed to be enforced consistently. Uploads could partially succeed and partially fail. Cleanup operations could remove some resources while leaving others behind. Conversations needed guarantees around who could access them. The project was no longer being challenged by AI capabilities—it was being challenged by software engineering realities.

That became the focus of V3.

V3 was less about adding features and more about making the system trustworthy.

One of the biggest architectural changes was moving to a contract-first approach between the FastAPI backend and Next.js frontend. Instead of manually maintaining API types, the frontend now consumes TypeScript types generated directly from the backend's OpenAPI schema. This reduced duplication and helped prevent frontend-backend drift as the application evolved.

I also redesigned authentication boundaries. Earlier versions relied more heavily on client-supplied information, but V3 shifted toward server-derived identity. Backend authorization now depends on identity asserted by trusted server infrastructure rather than user identifiers sent from the browser. This forced me to think more seriously about trust boundaries and ownership enforcement.

Another major improvement was introducing explicit lifecycle management.

Uploads and deletions are no longer treated as simple success-or-failure actions. Instead, the system tracks where failures occur, whether during validation, storage, metadata persistence, indexing, or cleanup. Delete operations support partial-cleanup outcomes, making them safer and easier to recover from when something goes wrong.

At the same time, I began focusing on answer quality.

Most RAG tutorials stop once retrieval returns text and the model generates a response. I wanted stronger guarantees.

I introduced structured answer generation through a JSON-based response contract. The system now returns metadata such as retrieval mode, intent classification, answer status, and citations alongside the generated answer. Retrieval policies differ depending on the user's intent, and generated answers pass through evidence-aware validation before being returned. If supporting evidence is weak or grounding checks fail, the system explicitly falls back to an insufficient-context response rather than confidently generating unsupported answers.

This was one of the most important lessons of the project:

> A trustworthy AI application is defined as much by what it refuses to answer as by what it answers correctly.

To support these guarantees, I invested heavily in testing.

V3 introduced backend integration coverage for authorization, ownership enforcement, lifecycle operations, retrieval policies, API contracts, and vector-store behavior. On the frontend, I added tests covering document selection, uploads, deletions, workspace transitions, recovery scenarios, and protection against stale asynchronous state updates.

What began as a PDF chatbot had evolved into a system with authentication boundaries, persistence layers, ownership rules, lifecycle management, structured contracts, reliability guarantees, and testing infrastructure.

Today, I don't view AskMyDoc as a document chatbot.

I view it as a record of my growth as an engineer.

V1 taught me how RAG systems work.

V2 taught me how AI features become products.

V3 taught me how products become reliable systems.

The most valuable lessons weren't about embeddings, vector databases, or prompting. They were about contracts, trust boundaries, failure handling, authorization, testing, and building software that behaves predictably when things go wrong.

AskMyDoc started as a way to learn RAG.

It eventually became a project that taught me how to think about engineering AI products as complete systems rather than isolated AI features.

During the Markdown upload work, I ran into a real deployment issue that wasn't caused by the extraction or indexing code itself. Markdown uploads initially failed at the storage layer because the Supabase bucket was configured to allow only `application/pdf`, so uploads tagged as `text/markdown` or `text/plain` were rejected with `415 invalid_mime_type`.

The fix was to update the `pdfs` storage bucket configuration to allow the MIME types needed by the shared upload flow, specifically `application/pdf`, `text/markdown`, and `text/plain`. Once the bucket configuration matched the application behavior, Markdown uploads worked correctly and empty Markdown files were still rejected cleanly by the validation layer.
