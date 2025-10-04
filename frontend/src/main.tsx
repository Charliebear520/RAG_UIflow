import React from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap-icons/font/bootstrap-icons.css";
import "./styles.css";
import { Layout } from "./routes/Layout";
import { UploadPage } from "./routes/UploadPage";
import { ChunkPage } from "./routes/ChunkPage";
import { EmbedPage } from "./routes/EmbedPage";
import { RetrievePage } from "./routes/RetrievePage";
import { EnhancedRetrievePage } from "./routes/EnhancedRetrievePage";
import { GeneratePage } from "./routes/GeneratePage";
import { EvaluatePage } from "./routes/EvaluatePage";
import { RagProvider } from "./lib/ragStore";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <UploadPage /> },
      { path: "chunk", element: <ChunkPage /> },
      { path: "embed", element: <EmbedPage /> },
      { path: "retrieve", element: <RetrievePage /> },
      { path: "enhanced-retrieve", element: <EnhancedRetrievePage /> },
      { path: "generate", element: <GeneratePage /> },
      { path: "evaluate", element: <EvaluatePage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RagProvider>
      <RouterProvider router={router} />
    </RagProvider>
  </React.StrictMode>
);
