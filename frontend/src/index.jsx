import React from "react";
import ReactDOM from "react-dom/client";
import { Streamlit } from "streamlit-component-lib";
import FloorplanComponent from "./wallEditor";

import "./styles.css";

const container = document.getElementById("root");

if (container) {
  const root = ReactDOM.createRoot(container);
  root.render(
    <React.StrictMode>
      <FloorplanComponent />
    </React.StrictMode>
  );
}

Streamlit.setComponentReady();
Streamlit.setFrameHeight(700);
